"""Fail-closed normal/debug observability contract.

This adapter deliberately preserves the existing ``stage_timing`` profile
files.  It normalizes their legacy fields for new callers without changing
the running experiment or the on-disk event schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class ObservabilityProfileError(ValueError):
    """Raised when an observability profile is ambiguous or unsafe."""


@dataclass(frozen=True)
class ObservabilityProfile:
    mode: str
    sample_timing: bool
    node_timing: bool
    trace_payloads: bool
    resource_metrics: bool
    flush_each_event: bool
    monitor_interval_seconds: int


def parse_profile(config: Mapping[str, object] | None) -> ObservabilityProfile:
    """Normalize the pinned legacy profile without silently accepting typos.

    ``None`` selects the production normal profile.  The repository's current
    files use ``mode``, ``client_detail`` and
    ``server_metrics_interval_seconds``; those names remain authoritative.
    """
    if config is None:
        return ObservabilityProfile("normal", True, False, False, True, False, 600)
    if not isinstance(config, Mapping):
        raise ObservabilityProfileError("profile must be a mapping")
    allowed = {"mode", "client_detail", "server_metrics_interval_seconds"}
    if set(config) != allowed:
        raise ObservabilityProfileError("profile fields do not match the pinned legacy contract")
    mode = config.get("mode")
    detail = config.get("client_detail")
    interval = config.get("server_metrics_interval_seconds")
    if mode not in ("normal", "debug") or type(detail) is not bool:
        raise ObservabilityProfileError("invalid mode or client_detail")
    if detail is not (mode == "debug"):
        raise ObservabilityProfileError("client_detail does not match mode")
    if type(interval) is not int or interval < 10:
        raise ObservabilityProfileError("monitor interval must be an integer >= 10")
    if mode == "normal" and interval != 600:
        raise ObservabilityProfileError("normal mode must use a 600 second interval")
    return ObservabilityProfile(
        mode=mode,
        sample_timing=True,
        node_timing=detail,
        trace_payloads=detail,
        resource_metrics=True,
        flush_each_event=detail,
        monitor_interval_seconds=interval,
    )


__all__ = ["ObservabilityProfile", "ObservabilityProfileError", "parse_profile"]
