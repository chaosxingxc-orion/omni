"""CPU-only validation for a fixed real-data concurrency-sweep manifest.

This module never loads labels, contacts a model, or authorizes a flight. It
binds a small shared manifest to the pinned real roster and a deployment
profile so a later queue can use one identity for every concurrency level.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "saea-concurrency-sweep-workload-v1"
MANIFEST_SCHEMA = "saea-fixed-real-subset-manifest-v1"


class WorkloadError(ValueError):
    pass


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"cannot read JSON {path}: {exc}") from exc


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise WorkloadError(f"cannot hash {path}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkloadError(message)


def validate_manifest(
    manifest_path: Path,
    *,
    source_roster_path: Path,
    expected_profile_id: str,
    expected_profile_sha256: str,
    expected_count: int = 100,
) -> dict[str, Any]:
    """Validate an immutable real subset without opening labels or model code."""
    # Hash exactly the bytes that were validated, even if a path is replaced.
    manifest_bytes = manifest_path.read_bytes()
    roster_bytes = source_roster_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    roster = json.loads(roster_bytes)
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "fixed real manifest schema mismatch")
    status_ok = manifest.get("status") == "PINNED_CPU_ONLY_NOT_ADMITTED"
    legacy_selection_ok = manifest.get("selection") == "first-100-of-pinned-real-engine-500-v1; no labels/outcomes"
    _require(status_ok or legacy_selection_ok, "manifest is not held")
    _require(roster.get("schema") == "saea-real-engine-roster-v1", "source roster schema mismatch")
    items = manifest.get("items")
    source_items = roster.get("items")
    _require(isinstance(items, list) and len(items) == expected_count, "manifest row count mismatch")
    _require(isinstance(source_items, list), "source roster items missing")
    _require(manifest.get("count") == expected_count, "manifest count field mismatch")
    source_name = str(manifest.get("source_roster", ""))
    _require(source_name == "real-engine-500-v1" or source_name.endswith("/real-engine-500-v1/roster.json"), "unexpected source roster")
    _require(manifest.get("source_roster_sha256") == hashlib.sha256(roster_bytes).hexdigest(), "source roster hash mismatch")
    _require(manifest.get("test_labels_read", False) is False, "manifest admits test labels")
    _require(manifest.get("model_calls", 0) == 0, "manifest records model calls")
    by_id = {row.get("sample_id"): row for row in source_items}
    _require(len(by_id) == len(source_items), "source roster has duplicate sample IDs")
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    for row in items:
        _require(isinstance(row, dict), "manifest item is not an object")
        sid = row.get("sample_id")
        audio = row.get("audio_sha256")
        _require(isinstance(sid, str) and sid not in seen_ids, "manifest sample IDs are duplicate or invalid")
        _require(isinstance(audio, str) and len(audio) == 64 and audio not in seen_audio, "manifest audio hashes are duplicate or invalid")
        source = by_id.get(sid)
        _require(source is not None, f"manifest sample is absent from source roster: {sid}")
        for key in ("packet_id", "media_relpath", "audio_sha256", "audio_seconds", "sample_rate_hz"):
            _require(row.get(key) == source.get(key), f"manifest/source mismatch for {sid}: {key}")
        _require(str(row.get("media_relpath", "")).startswith("datasets/slurp/slurp_real/"), f"non-real media path: {sid}")
        seen_ids.add(sid)
        seen_audio.add(audio)
    _require(manifest.get("unique_sample_ids", expected_count) == expected_count, "unique sample count field mismatch")
    _require(manifest.get("unique_audio_hashes", expected_count) == expected_count, "unique audio count field mismatch")
    return {
        "schema": SCHEMA,
        "status": "HELD_NOT_ADMITTED",
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_roster_path": str(source_roster_path),
        "source_roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
        "profile_id": expected_profile_id,
        "profile_sha256": expected_profile_sha256,
        "rows": expected_count,
        "sample_ids": sorted(seen_ids),
        "items": items,
        "model_calls": 0,
        "test_labels_read": False,
        "generation_authorized": False,
        "admission": "held",
    }


def build_level_spec(
    validated: dict[str, Any], *, level_id: str, concurrency_per_instance: int, max_num_seqs: int = 128
) -> dict[str, Any]:
    """Create a queue input record; this function does not admit or launch it."""
    _require(validated.get("status") == "HELD_NOT_ADMITTED", "validated workload is not held")
    _require(level_id in {"C32", "C48", "C64", "C96"}, "unknown sweep level")
    _require(1 <= concurrency_per_instance <= max_num_seqs, "invalid concurrency level")
    return {
        "schema": "saea-concurrency-sweep-level-v1",
        "status": "PLAN_ONLY_NOT_ADMITTED",
        "level_id": level_id,
        "concurrency_per_instance": concurrency_per_instance,
        "max_num_seqs": max_num_seqs,
        "profile_id": validated["profile_id"],
        "profile_sha256": validated["profile_sha256"],
        "manifest_sha256": validated["manifest_sha256"],
        "rows": validated["rows"],
        "model_calls": 0,
        "test_labels_read": False,
        "generation_authorized": False,
    }
