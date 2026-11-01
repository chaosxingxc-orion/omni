import json

import pytest

from speech_aware_evidence_acquisition.data.runtime_manifest_cache import (
    RuntimeManifestError,
    build_manifest_from_engineering_gate,
    build_manifest,
    load_manifest,
    write_manifest,
)
from types import SimpleNamespace


def row(sample_id="s1"):
    return {
        "sample_id": sample_id,
        "media_relpath": f"audio/{sample_id}.flac",
        "audio_sha256": "a" * 64,
        "audio_seconds": 1.25,
        "sample_rate_hz": 16000,
    }


def test_manifest_is_sorted_and_reloads_with_source_binding(tmp_path):
    source = {"roster_sha256": "b" * 64, "split_identity": "split-v1"}
    document = build_manifest(source_identity=source, items=[row("s2"), row("s1")])
    path = tmp_path / "runtime-manifest.json"
    digest = write_manifest(path, document)
    loaded = load_manifest(path, source_identity=source)
    assert digest == loaded["manifest_sha256"]
    assert [item["sample_id"] for item in loaded["items"]] == ["s1", "s2"]


def test_partial_or_stale_manifest_is_refused(tmp_path):
    source = {"roster_sha256": "b" * 64}
    path = tmp_path / "runtime-manifest.json"
    write_manifest(path, build_manifest(source_identity=source, items=[row()]))
    document = json.loads(path.read_text())
    document["items"][0]["audio_seconds"] = 9.0
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(RuntimeManifestError, match="content hash"):
        load_manifest(path, source_identity=source)


def test_source_identity_and_runtime_fields_fail_closed():
    with pytest.raises(RuntimeManifestError, match="source identity"):
        build_manifest(source_identity={}, items=[row()])
    with pytest.raises(RuntimeManifestError, match="escapes"):
        build_manifest(
            source_identity={"roster_sha256": "b" * 64},
            items=[{**row(), "media_relpath": "../secret.flac"}],
        )


def test_engineering_gate_adapter_keeps_only_label_free_identity():
    gate = SimpleNamespace(
        split_identity_hash="b" * 64,
        runtime_view_sha256="c" * 64,
        samples=(SimpleNamespace(**row("s1"), packet_id="ignored"),),
    )
    manifest = build_manifest_from_engineering_gate(gate)
    assert manifest["count"] == 1
    assert "packet_id" not in manifest["items"][0]
    assert set(manifest["source_identity"]) == {"split_identity_hash", "runtime_view_sha256"}
