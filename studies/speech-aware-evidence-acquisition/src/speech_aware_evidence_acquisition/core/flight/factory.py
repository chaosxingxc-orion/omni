"""One four-fragment factory for carrier-scoped shared FlightRuntime workers."""
from __future__ import annotations

import inspect
import hashlib
import json
import os
from functools import partial
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from ...data.carriers import get_carrier_descriptor
from ...data.lock import load_lock, umbrella_lock_path
from ...rails.task_adapter import TaskAdapter
from ..base import TASK_ADAPTERS, CoreError, declared_derived_media_roots
from ..config import compose
from ..scoring_batch import ScoringContext, run_score_batch
from .runtime import FlightRuntime
from .runtime_overlay import apply_runtime_overlay


def load_external_runtime_config(path: Path | None) -> Mapping[str, object]:
    """Load an immutable operational overlay supplied by the launcher.

    Scientific fields remain in the four registered fragments.  The overlay
    is limited to scheduling/observability knobs and is copied into the
    composed values before runtime construction.  A caller may bind its raw
    SHA256 through ``SAEA_RUNTIME_CONFIG_SHA256``; a changed file then fails
    before any adapter, session, FIT or model is touched.
    """
    if path is None:
        return {}
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    expected = os.environ.get("SAEA_RUNTIME_CONFIG_SHA256")
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and expected != actual:
        raise CoreError("external runtime config SHA256 differs from launcher pin")
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, Mapping):
        raise CoreError("external runtime config must be a JSON object")
    schema = document.get("schema")
    if schema not in {"saea-request-pipeline-v1", "saea-runtime-efficiency-v1", "saea-native-efficiency-v1"}:
        raise CoreError("unsupported external runtime config schema")
    scheduler = "dispatch" if schema == "saea-native-efficiency-v1" else "request_pipeline"
    overlay = document.get(scheduler)
    if not isinstance(overlay, Mapping):
        raise CoreError(f"external runtime config must declare {scheduler}")
    if scheduler == "dispatch":
        from ...rails.dispatch_slot_fill import DispatchSlotFillError, parse_dispatch_slot_fill
        try:
            parse_dispatch_slot_fill({"dispatch": overlay})
        except DispatchSlotFillError as error:
            raise CoreError(str(error)) from error
    allowed = {"schema", scheduler}
    if schema == "saea-native-efficiency-v1" and "native_fit" in document:
        allowed.add("native_fit")
        fit = document["native_fit"]
        if (not isinstance(fit, Mapping) or set(fit) != {"max_concurrency"}
                or type(fit["max_concurrency"]) is not int
                or fit["max_concurrency"] not in (8, 16, 32)):
            raise CoreError("native_fit.max_concurrency must be 8, 16 or 32")
    if schema in {"saea-runtime-efficiency-v1", "saea-native-efficiency-v1"}:
        allowed.add("audio_cache")
        cache = document.get("audio_cache")
        if not isinstance(cache, Mapping):
            raise CoreError("runtime efficiency config must declare audio_cache")
        if set(cache) != {"enabled", "max_entries", "max_bytes", "preload_workers"}:
            raise CoreError("audio_cache must declare enabled, max_entries, max_bytes and preload_workers")
        if not isinstance(cache["enabled"], bool):
            raise CoreError("audio_cache.enabled must be a boolean")
        for key, upper in (("max_entries", 1_000_000), ("max_bytes", 64 * 1024 * 1024 * 1024), ("preload_workers", 64)):
            value = cache[key]
            if type(value) is not int or not 0 <= value <= upper or (key == "preload_workers" and value < 1):
                raise CoreError(f"audio_cache.{key} is outside its allowed range")
    if set(document) != allowed:
        raise CoreError("external runtime config has unexpected fields")
    result = {
        scheduler: dict(overlay),
        "_external_runtime_config": {"path": str(path), "sha256": actual},
    }
    if "audio_cache" in document:
        result["audio_cache"] = dict(document["audio_cache"])
    if "native_fit" in document:
        result["native_fit"] = dict(document["native_fit"])
    return result


def register_builtin_task_adapters() -> None:
    from ...rails.adapters.slurp_runtime import register_runtime_adapter as register_slurp_runtime_adapter
    from ...rails.adapters.earnings_runtime import register_runtime_adapter as register_earnings_runtime_adapter
    from ...rails.adapters.audio2tool_runtime import register_runtime_adapter as register_audio2tool_runtime_adapter
    from ...rails.adapters.earnings22_runtime import register_runtime_adapter as register_earnings22_runtime_adapter
    register_slurp_runtime_adapter()
    register_earnings_runtime_adapter()
    register_audio2tool_runtime_adapter()
    register_earnings22_runtime_adapter()


def build_flight_runtime(
    repo_root: Path, data_root: Path, *, model: str, dataset: str, baseline: str,
    experiment: str, run_id: str, attempt_id: str, runtime_root: Path,
    lock_path: Path | None = None, lock: Mapping[str, object] | None = None,
    campaign_active: bool = True, **runtime_seams,
) -> FlightRuntime:
    """Resolve one carrier adapter; all arms keep the same controller and executor."""
    repo_root, data_root = Path(repo_root), Path(data_root)
    composed = compose(repo_root, model, dataset, baseline, experiment, data_root=data_root)
    overlay_path = runtime_seams.pop("runtime_config_path", None)
    overlay = load_external_runtime_config(overlay_path)
    if overlay:
        values = apply_runtime_overlay(composed.values, overlay, error_type=CoreError)
        composed = type(composed)(
            values=values,
            config_hash=hashlib.sha256(
                json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(), fragments=composed.fragments,
        )
    values = composed.values
    descriptor = get_carrier_descriptor(str(values["carrier_lock_key"]))
    register_builtin_task_adapters()
    builder = TASK_ADAPTERS.resolve(descriptor.lock_key)
    resolved_lock = lock if lock is not None else load_lock(
        Path(lock_path) if lock_path is not None else umbrella_lock_path(repo_root)
    )
    # A builder that declares ``run_id`` receives the flight's own run id: the
    # SLURP adapter binds a final-report test-sample view to the owner
    # allowance's permitted run ids. Every other builder keeps its signature.
    builder_kwargs = {"dataset_fragment": dataset}
    if "run_id" in inspect.signature(builder).parameters:
        builder_kwargs["run_id"] = run_id
    adapter = builder(values, resolved_lock, data_root, repo_root, **builder_kwargs)
    if not isinstance(adapter, TaskAdapter):
        raise CoreError("registered task adapter builder must return TaskAdapter")
    url = urlparse(str(values.get("base_url", "http://127.0.0.1:8080")))
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise CoreError("flight workers require the loopback resident service")
    options = {"host": url.hostname, "port": url.port or 8080}
    # Deployment is operational metadata, outside the composed protocol. Both the
    # lease CLI and campaign factory must see the same explicit host selection.
    deployment = os.environ.get("SAEA_DEPLOYMENT_PROFILE")
    if deployment:
        if values.get("transport") != "vllm":
            raise CoreError("SAEA_DEPLOYMENT_PROFILE requires a vllm model fragment")
        options["deployment_profile"] = deployment
    admitted = os.environ.get("SAEA_ADMITTED_LEVEL")
    if admitted is not None:
        if not deployment:
            raise CoreError("SAEA_ADMITTED_LEVEL requires SAEA_DEPLOYMENT_PROFILE")
        try:
            options["admitted_level"] = int(admitted)
        except ValueError as error:
            raise CoreError("SAEA_ADMITTED_LEVEL must be an integer") from error
    roots = declared_derived_media_roots(values, data_root)
    if len(roots) > 1:
        raise CoreError("runtime carrier view must select one derived media root")
    if roots:
        options["carrier_root_resolver"] = lambda *_args: roots[0]
    if lock_path is not None:
        options["score_batch"] = partial(
            run_score_batch,
            context=ScoringContext(repo_root, data_root, Path(lock_path)),
        )
    options.update(runtime_seams)
    runtime = FlightRuntime(
        repo_root, data_root, model=model, dataset=dataset, baseline=baseline,
        experiment=experiment, run_id=run_id, attempt_id=attempt_id,
        runtime_root=runtime_root, adapter_factory=lambda *_args: adapter,
        lock=resolved_lock, campaign_active=campaign_active,
        runtime_config_overlay=overlay if overlay else None,
        **options,
    )
    adapter.bind_runtime(runtime, values)
    return runtime
