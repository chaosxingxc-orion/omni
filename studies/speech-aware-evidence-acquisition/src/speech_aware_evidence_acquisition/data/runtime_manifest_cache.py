"""Persistent, label-free runtime manifests shared by independent flights.

The manifest is a CPU-side preparation artifact.  It contains only media
identity and header facts, never transcript, annotation, answer, or prompt
text.  A producer may build it once from a frozen roster; later experiments
can load the same bytes without rescanning the dataset.  Reads are fail-closed
and writes are atomic so an interrupted preparation cannot publish a partial
manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "saea-runtime-manifest-cache-v1"
_SHA256 = set("0123456789abcdef")
_REQUIRED_ITEM_KEYS = frozenset(
    {"sample_id", "media_relpath", "audio_sha256", "audio_seconds", "sample_rate_hz"}
)


class RuntimeManifestError(ValueError):
    """The manifest is malformed, stale, or not safely bound to its source."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _SHA256 for ch in value):
        raise RuntimeManifestError(f"{label} must be a lowercase SHA256")
    return value


def _item(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != _REQUIRED_ITEM_KEYS:
        raise RuntimeManifestError(f"manifest item {index} has unexpected fields")
    sample_id = item["sample_id"]
    relpath = item["media_relpath"]
    if not isinstance(sample_id, str) or not sample_id:
        raise RuntimeManifestError(f"manifest item {index} has an invalid sample_id")
    if not isinstance(relpath, str) or not relpath or "\\" in relpath:
        raise RuntimeManifestError(f"manifest item {index} has an invalid media_relpath")
    path = Path(relpath)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeManifestError(f"manifest item {index} media_relpath escapes its data root")
    seconds = item["audio_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise RuntimeManifestError(f"manifest item {index} audio_seconds must be positive and finite")
    rate = item["sample_rate_hz"]
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise RuntimeManifestError(f"manifest item {index} sample_rate_hz must be positive")
    return {
        "sample_id": sample_id,
        "media_relpath": relpath,
        "audio_sha256": _sha(item["audio_sha256"], f"manifest item {index} audio_sha256"),
        "audio_seconds": float(seconds),
        "sample_rate_hz": rate,
    }


def _items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or not items:
        raise RuntimeManifestError("manifest items must be a non-empty sequence")
    normalized = [_item(item, i) for i, item in enumerate(items)]
    ids = [item["sample_id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise RuntimeManifestError("manifest sample_id values must be unique")
    normalized.sort(key=lambda item: item["sample_id"])
    return normalized


def build_manifest(*, source_identity: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build canonical cache bytes from a frozen source identity and media rows."""

    if not isinstance(source_identity, Mapping) or not source_identity:
        raise RuntimeManifestError("source_identity must be a non-empty mapping")
    normalized = _items(items)
    unsigned = {
        "schema": SCHEMA,
        "source_identity": dict(source_identity),
        "source_identity_sha256": _digest(dict(source_identity)),
        "count": len(normalized),
        "items": normalized,
    }
    return {**unsigned, "manifest_sha256": _digest(unsigned)}


def build_manifest_from_engineering_gate(gate: Any) -> dict[str, Any]:
    """Build a shared manifest directly from a verified label-free gate.

    ``load_slurp_engineering_gate`` has already checked media bytes and audio
    headers.  This adapter deliberately reads only its identity fields and
    samples, making the resulting cache safe to share with every later flight
    without reopening the annotation layer.
    """

    try:
        split_identity = gate.split_identity_hash
        view_identity = gate.runtime_view_sha256
        samples = gate.samples
    except AttributeError as error:
        raise RuntimeManifestError("engineering gate lacks the verified identity fields") from error
    source_identity = {
        "split_identity_hash": _sha(split_identity, "engineering gate split identity"),
        "runtime_view_sha256": _sha(view_identity, "engineering gate runtime view identity"),
    }
    items = []
    for sample in samples:
        try:
            items.append(
                {
                    "sample_id": sample.sample_id,
                    "media_relpath": sample.media_relpath,
                    "audio_sha256": sample.audio_sha256,
                    "audio_seconds": sample.audio_seconds,
                    "sample_rate_hz": sample.sample_rate_hz,
                }
            )
        except AttributeError as error:
            raise RuntimeManifestError("engineering gate sample lacks media identity fields") from error
    return build_manifest(source_identity=source_identity, items=items)


def validate_manifest(document: Any, *, source_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a loaded document and return a detached normalized copy."""

    if not isinstance(document, Mapping):
        raise RuntimeManifestError("runtime manifest must be an object")
    required = {"schema", "source_identity", "source_identity_sha256", "count", "items", "manifest_sha256"}
    if set(document) != required or document.get("schema") != SCHEMA:
        raise RuntimeManifestError("runtime manifest schema or fields are invalid")
    identity = document["source_identity"]
    if not isinstance(identity, Mapping) or not identity:
        raise RuntimeManifestError("runtime manifest source identity is invalid")
    if document["source_identity_sha256"] != _digest(dict(identity)):
        raise RuntimeManifestError("runtime manifest source identity hash mismatch")
    if source_identity is not None and dict(identity) != dict(source_identity):
        raise RuntimeManifestError("runtime manifest source identity differs from requested source")
    normalized = _items(document["items"])
    if document["count"] != len(normalized):
        raise RuntimeManifestError("runtime manifest count mismatch")
    unsigned = {
        "schema": SCHEMA,
        "source_identity": dict(identity),
        "source_identity_sha256": document["source_identity_sha256"],
        "count": len(normalized),
        "items": normalized,
    }
    if document["manifest_sha256"] != _digest(unsigned):
        raise RuntimeManifestError("runtime manifest content hash mismatch")
    return {**unsigned, "manifest_sha256": document["manifest_sha256"]}


def write_manifest(path: Path, document: Mapping[str, Any]) -> str:
    """Atomically publish a validated manifest and return its content hash."""

    checked = validate_manifest(document)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(checked) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(str(target.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return checked["manifest_sha256"]


def load_manifest(path: Path, *, source_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load canonical bytes and refuse a partial, edited, or stale cache."""

    target = Path(path)
    try:
        raw = target.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeManifestError(f"cannot read runtime manifest {target}: {error}") from error
    if raw != _canonical(document) + b"\n":
        raise RuntimeManifestError("runtime manifest is not canonical JSON")
    return validate_manifest(document, source_identity=source_identity)
