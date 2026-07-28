"""Streaming inference service Deployment implementation.

Provides streaming inference capability based on SSE (Server-Sent Events),
suitable for scenarios where LLM tokens are returned one by one.

Design highlights:
- Abstract base class ``StreamingInferenceService`` is not decorated with ``@serve.deployment``;
  concrete subclasses (e.g., ``LLMStreamingService``) apply the decorator.
- Synchronous inference is wrapped via a custom thread pool to avoid blocking the asyncio event loop.
- Request/response follows OpenAI Chat Completions API format.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from starlette.requests import Request
from starlette.responses import StreamingResponse

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class StreamingInferenceService(ABC):
    """Streaming inference service base class.

    Not directly decorated with ``@serve.deployment``; concrete subclasses apply the decorator.
    Provides common logic such as SSE response wrapping, connection lifecycle management,
    and error handling.
    """

    def __init__(self) -> None:
        """Initialize streaming inference service."""
        self._model_loaded = False

    @abstractmethod
    async def _generate_stream(
        self, input_data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response, must be implemented by subclasses.

        Args:
            input_data: Parsed request data.

        Yields:
            JSON formatted data chunks (without SSE prefix).
        """

    async def __call__(self, request: Request) -> StreamingResponse:
        """Handle streaming inference request."""
        try:
            data = await request.json()
        except Exception:
            return StreamingResponse(
                iter(['data: {"error": "Invalid JSON"}\n\n']),
                media_type="text/event-stream",
                status_code=400,
            )

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in self._generate_stream(data):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                logger.info("Client disconnected, cleaning up...")
                raise
            except Exception as e:
                logger.error("Streaming error: %s", e, exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def health(self) -> dict[str, Any]:
        """Health check."""
        return {
            "status": "healthy" if self._model_loaded else "unhealthy",
            "model_loaded": self._model_loaded,
        }


@PublicAPI(stability="beta")
class LLMStreamingService(StreamingInferenceService):
    """LLM streaming inference service.

    Inherits from ``StreamingInferenceService``, implementing LLM-specific streaming generation logic.
    Uses ``ThreadPoolExecutor`` to execute synchronous inference in threads, avoiding blocking the event loop;
    Model loading is async via ``asyncio.to_thread()`` to avoid blocking the event loop in the constructor.

    Args:
        model_path: Model file path.
        tokenizer_path: Tokenizer file path.
        max_tokens: Default maximum number of tokens to generate.
        max_workers: Inference thread pool size.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_tokens: int = 512,
        max_workers: int = 4,
    ) -> None:
        super().__init__()
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._max_tokens = max_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="llm-inference",
        )
        self._model_loaded = False

    async def reconfigure(self, user_config: Any) -> None:
        """Ray Serve lifecycle hook: asynchronously loads the model.

        Called after ``__init__`` to avoid blocking the replica's event loop with synchronous loading.
        """
        if not self._model_loaded:
            await asyncio.to_thread(self._load_model_sync)

    async def health(self) -> dict[str, Any]:
        """Health check, verifies that both model and tokenizer are loaded."""
        healthy = (
            self._model_loaded
            and self._model is not None
            and self._tokenizer is not None
        )
        return {
            "status": "healthy" if healthy else "unhealthy",
            "model_loaded": self._model_loaded,
        }

    def __del__(self) -> None:
        """Shutdown ThreadPoolExecutor to prevent thread leaks when replica is destroyed.

        Ray Serve's ``call_destructor()`` only shuts down its internal thread pool and does not touch
        user-created executors. ``wait=False`` is used here to avoid blocking the destructor; already
        started forward threads cannot be safely interrupted by Python and will be forcefully terminated
        by Ray when the replica process exits.
        """
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _build_prompt_from_messages(messages: list[dict[str, str]]) -> str:
        """Build prompt when the tokenizer does not have ``apply_chat_template``.

        This fallback cannot guarantee exact match with the model's training format;
        it only serves as a reasonable approximation when no chat template is available.
        """
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nAssistant:"

    def _load_model_sync(self) -> None:
        """Synchronously load the LLM model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)

            has_cuda = torch.cuda.is_available()
            dtype = torch.float16 if has_cuda else torch.float32
            device_map = "auto" if has_cuda else None

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
            if not has_cuda:
                self._model = self._model.to("cpu")
            self._model.eval()
            self._model_loaded = True
            logger.info(
                "LLM model loaded from %s (device=%s, dtype=%s)",
                self._model_path,
                self._model.device,
                dtype,
            )
        except Exception as e:
            logger.error("Failed to load LLM model: %s", e)
            self._model_loaded = False
            raise

    async def _generate_stream(
        self, input_data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """LLM token streaming generation.

        Reuses ``past_key_values`` to avoid redundant KV-cache computation; uses ``threading.Event``
        to propagate client disconnect signals, allowing underlying inference threads to exit promptly.
        """
        import torch

        # Parse request (supports OpenAI format and simplified format)
        if "messages" in input_data:
            messages = input_data["messages"]
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self._build_prompt_from_messages(messages)
        else:
            prompt = input_data.get("prompt", "")

        max_tokens = input_data.get("max_tokens", self._max_tokens)
        temperature = input_data.get("temperature", 0.7)

        if not self._model_loaded:
            yield json.dumps({"error": "Model not loaded"})
            return

        cancel_event = threading.Event()
        loop = asyncio.get_event_loop()
        device = self._model.device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            input_ids.shape[-1], dtype=torch.long, device=device
        ).unsqueeze(0)
        past_key_values: Any = None

        try:
            for _ in range(max_tokens):
                (
                    past_key_values,
                    token_id,
                    attention_mask,
                    position_ids,
                ) = await loop.run_in_executor(
                    self._executor,
                    self._generate_one_token,
                    self._model,
                    self._tokenizer,
                    input_ids,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    temperature,
                    cancel_event,
                )

                token_text = self._tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                chunk = {
                    "choices": [
                        {
                            "delta": {"content": token_text},
                            "finish_reason": None,
                        }
                    ]
                }
                yield json.dumps(chunk)

                if token_id == self._tokenizer.eos_token_id:
                    break

                input_ids = torch.tensor([[token_id]], device=device)
        except asyncio.CancelledError:
            # asyncio.CancelledError can only be propagated cooperatively: set the signal so the next
            # _generate_one_token exits proactively. The currently executing forward thread
            # cannot be safely interrupted by Python; this is an inherent limitation of concurrent.futures.
            cancel_event.set()
            logger.info("Client disconnected, cancelling generation.")
            raise
        finally:
            cancel_event.set()

        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})

    @staticmethod
    def _generate_one_token(
        model: Any,
        tokenizer: Any,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        past_key_values: Any,
        temperature: float,
        cancel_event: threading.Event,
    ) -> tuple[Any, int, Any, Any]:
        """Execute single-step forward in a thread, reusing ``past_key_values``.

        Args:
            model: Loaded Causal LM.
            tokenizer: Tokenizer (reserved for future extension).
            input_ids: Current input token ids (single token at the last step).
            attention_mask: Attention mask for the current sequence.
            position_ids: Position ids for the current sequence.
            past_key_values: Cached KV states from the previous step.
            temperature: Sampling temperature; ``<= 0`` uses greedy decoding.
            cancel_event: Cancellation signal; if set, raises an exception to interrupt generation.

        Returns:
            Updated ``past_key_values``, predicted next token id,
            updated ``attention_mask`` and ``position_ids``.
        """
        import torch

        if cancel_event.is_set():
            raise RuntimeError("Generation cancelled")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token_id = torch.argmax(logits, dim=-1).item()

        next_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=-1,
        )
        next_position_ids = position_ids[:, -1:] + 1

        return (
            outputs.past_key_values,
            next_token_id,
            next_attention_mask,
            next_position_ids,
        )
