import hashlib
import json

import pytest

from speech_aware_evidence_acquisition.rails.tfgrpo_learning import LearningError
from speech_aware_evidence_acquisition.rails.tfgrpo_verified import VerifiedImportContext


def pin(path):
    return {"relpath": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_context_reuses_parsed_document_and_returns_isolated_copy(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"value": [1]}))
    context = VerifiedImportContext(data_root=tmp_path)
    reads = []
    original = path.read_bytes

    def read(current):
        if current == path:
            reads.append(1)
        return original()

    monkeypatch.setattr(type(path), "read_bytes", read)
    first, doc = context.read(pin(path))
    doc["value"].append(2)
    second, again = context.read(pin(path))
    assert first == second == path
    assert again == {"value": [1]}
    assert len(reads) == 4  # two pin reads plus initial load and hit digest check


def test_context_rejects_replaced_bytes_even_when_pin_is_old(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"value": 1}))
    source_pin = pin(path)
    context = VerifiedImportContext(data_root=tmp_path)
    context.read(source_pin)
    path.write_text(json.dumps({"value": 2}))
    with pytest.raises(LearningError, match="changed"):
        context.read(source_pin)


def test_context_rejects_missing_source_after_cached_read(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"value": 1}))
    source_pin = pin(path)
    context = VerifiedImportContext(data_root=tmp_path)
    context.read(source_pin)
    path.unlink()
    with pytest.raises(LearningError, match="disappeared"):
        context.read(source_pin)
