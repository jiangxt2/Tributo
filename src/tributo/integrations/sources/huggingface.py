"""HuggingFace model → ExportSource provider.

Resolves a HuggingFace model id (or a local ``(model, tokenizer)`` pair)
into an ``ExportSource`` for the export pipeline.  The model object and
its ``config.json`` are carried into the source so exporters can attach
the tokenizer/config to their artifacts (plan: "HF artifact 包含
tokenizer/config").
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from typing import Any, ClassVar, Generator

from pydantic import BaseModel, ConfigDict

from tributo.exporting.models import ExportSource
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class _HuggingFaceSourceOptions(BaseModel):
    """Options for the HuggingFace source provider."""

    model_config = ConfigDict(extra="forbid")

    task: str | None = None
    trust_remote_code: bool = False
    max_length: int | None = None


@PublicAPI(stability="beta")
class HuggingFaceSourceProvider:
    """Resolve a HuggingFace model to an ``ExportSource``.

    ``result`` is either:
    - a model id string (``"bert-base-uncased"``) or a local directory, or
    - a ``(model, tokenizer)`` tuple from a prior in-process load.
    """

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "hf-v1"
    trainer_type: ClassVar[str] = "hf"
    priority: ClassVar[int] = 100

    def open_source(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Any:
        """Open a HuggingFace model as an ``ExportSource``.

        Args:
            result: HF model id, local directory, or ``(model, tokenizer)``.
            config: Optional typed options (``_HuggingFaceSourceOptions``).

        Yields:
            ExportSource with source_kind="hf_model".
        """
        return self._open(result, config)

    @contextmanager
    def _open(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Generator[ExportSource, None, None]:
        import transformers

        opts = _HuggingFaceSourceOptions.model_validate(
            config.model_dump() if config is not None else {}
        )

        if isinstance(result, tuple) and len(result) == 2:
            model, tokenizer = result
            model_id = getattr(model, "name_or_path", "") or "unknown"
            model_config_data = (
                model.config.to_dict() if hasattr(model, "config") else {}
            )
            task = opts.task or model_config_data.get("task_type")
        else:
            # Model id or local directory — load with the task-aware class
            # when the task is known, otherwise the base AutoModel.
            model_id = str(result)
            task = opts.task
            if task and task.startswith("text-classification"):
                model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_id, trust_remote_code=opts.trust_remote_code
                )
            else:
                model = transformers.AutoModel.from_pretrained(
                    model_id, trust_remote_code=opts.trust_remote_code
                )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            model_config_data = model.config.to_dict()

        # Source fingerprint: config digest (stable across loads).
        fingerprint = hashlib.sha256(
            json.dumps(model_config_data, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        source = ExportSource(
            source_kind="hf_model",
            model_object=model,
            architecture_id=model_config_data.get("model_type"),
            model_config_data=model_config_data,
            metadata={
                "framework": "transformers",
                "framework_version": transformers.__version__,
                "model_id": model_id,
                "task": task,
                # The tokenizer is carried in-process so exporters can attach
                # it to their artifact (plan: HF artifact includes tokenizer).
                "tokenizer": tokenizer,
                "preprocessor": tokenizer,
            },
            source_fingerprint=fingerprint,
        )
        yield source
