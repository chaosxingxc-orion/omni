"""Apply operational controls identically at adapter and flight construction."""
from __future__ import annotations

import copy
from typing import Mapping


def apply_runtime_overlay(values, overlay, *, error_type=ValueError):
    """Copy explicit controls; never silently override a registered fragment.

    The caller recomputes the effective config hash. Operational changes still
    need a new attempt and the usual checkpoint compatibility proof.
    """
    result = dict(values)
    controls = ("request_pipeline", "dispatch", "audio_cache")
    if not isinstance(overlay, Mapping) or set(overlay) - {
        *controls, "_external_runtime_config",
    }:
        raise error_type("external runtime config has unexpected fields")
    schedulers = set(overlay) & {"request_pipeline", "dispatch"}
    if len(schedulers) != 1:
        raise error_type("external runtime config requires exactly one scheduler")
    for key in controls:
        if key not in overlay:
            continue
        if key in result:
            raise error_type(f"external runtime config collides with scientific {key}")
        if not isinstance(overlay[key], Mapping):
            raise error_type(f"external runtime config {key} is invalid")
        result[key] = copy.deepcopy(dict(overlay[key]))
    if "dispatch" in overlay:
        from ...rails.dispatch_slot_fill import DispatchSlotFillError, parse_dispatch_slot_fill

        try:
            dispatch = parse_dispatch_slot_fill(result)
        except DispatchSlotFillError as error:
            raise error_type(str(error)) from error
        if not isinstance(result.get("native_episode"), Mapping) or result.get("policy") == "fixed":
            raise error_type("external dispatch requires a native role episode")
        if "request_pipeline" in result or any(key in result for key in (
            "native_prefix_slots", "native_node_batch", "native_node_pipeline",
        )):
            raise error_type("external dispatch conflicts with another scheduler")
        if dispatch.ready_request and result.get("transport") != "vllm":
            raise error_type("ready_request requires vllm transport")
        if (dispatch.admitted_trajectories and dispatch.model_inflight
                and dispatch.admitted_trajectories < dispatch.model_inflight):
            raise error_type("admitted_trajectories must cover model_inflight")
    elif overlay["request_pipeline"].get("enabled") and "native_episode" in result:
        raise error_type("native episodes require dispatch, not request_pipeline")
    identity = overlay.get("_external_runtime_config", {})
    if not isinstance(identity, Mapping):
        raise error_type("external runtime config identity is invalid")
    result["runtime_config_identity"] = copy.deepcopy(dict(identity))
    return result
