"""Broker-specific CLI commands, mounted by :mod:`tributo.cli`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from tributo.exceptions import JobConfigurationError
from tributo.integrations.broker_registry import BrokerRegistry


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        raise click.ClickException("YAML broker config is not supported; use JSON.")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Unable to read broker config: {exc}") from exc
    if not isinstance(value, dict):
        raise click.ClickException("Broker config root must be a JSON object.")
    return value


@click.group()
def broker() -> None:
    """Discover and validate explicitly selected broker providers."""


@broker.command("list")
def broker_list() -> None:
    """List installed broker plugins without connecting to a broker."""
    registry = BrokerRegistry()
    descriptors = registry.list()
    for descriptor in descriptors:
        capabilities = ",".join(descriptor.capabilities) or "-"
        click.echo(
            f"{descriptor.broker_id}\tapi={descriptor.api_version}"
            f"\tstability={descriptor.stability}\tcapabilities={capabilities}"
        )
    for diagnostic in registry.diagnostics():
        click.echo(
            f"diagnostic\t{diagnostic.entry_point_name}\t{diagnostic.reason}",
            err=True,
        )


@broker.command("validate")
@click.option("--broker", "broker_id", required=True)
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--check-connectivity",
    is_flag=True,
    help="Ask the provider to perform an explicit connectivity probe.",
)
def broker_validate(broker_id: str, config_path: str, check_connectivity: bool) -> None:
    """Validate provider-owned JSON config."""
    try:
        config = _load_config(config_path)
        BrokerRegistry().validate(
            broker_id,
            config,
            check_connectivity=check_connectivity,
        )
    except (JobConfigurationError, click.ClickException) as exc:
        if isinstance(exc, click.ClickException):
            raise
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Broker configuration is valid: {broker_id}")
