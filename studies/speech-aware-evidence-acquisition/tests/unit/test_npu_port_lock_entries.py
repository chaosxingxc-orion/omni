"""The npu-port umbrella-lock amendment (owner decision 9.4, 2026-10-28) as committed in its readiness record.

The umbrella ``docs/datasets.lock.json`` is not versioned on the Ascend hosts, so the record
``docs/readiness/2026-10-28-npu-port-lock-entries-v1/`` (entries + apply script) is the reviewable form of the edit.
These checks keep that record self-consistent and shaped the way the lock's readers need it: the gate's model-entry
lookup (``local_subdir``, integer ``files`` and ``size_bytes``) and the port model fragment's ``model_lock_key``.
No asset bytes are read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.data.lock import lock_entry, load_lock

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD = REPO_ROOT / "docs/readiness/2026-10-28-npu-port-lock-entries-v1"
PORT_FRAGMENT = (
    REPO_ROOT / "configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1.json"
)
LIVE_LOCK = REPO_ROOT.parent.parent / "docs" / "datasets.lock.json"
_HEX64 = re.compile(r"[0-9a-f]{64}")
VIEWS = ("omni-embed-nemotron-3b-causal-v1", "lco-embedding-omni-3b-2605-v1", "lco-embedding-omni-7b-v1")


def _apply_module():
    spec = importlib.util.spec_from_file_location("npu_port_apply_amendment", RECORD / "apply_amendment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def apply():
    return _apply_module()


@pytest.fixture(scope="module")
def entries(apply):
    return apply.load_entries()  # refuses unless entries.json is the reviewed bytes


def test_record_pins_are_well_formed(apply):
    for pin in (apply.BASE_LOCK_SHA256, apply.ENTRIES_SHA256, apply.AMENDED_LOCK_SHA256):
        assert _HEX64.fullmatch(pin)
    assert len({apply.BASE_LOCK_SHA256, apply.AMENDED_LOCK_SHA256}) == 2


def test_every_entry_has_the_shape_the_gate_resolves(entries):
    names = [entry["name"] for entry in entries["models"]]
    assert names == ["qwen3-omni-30b-a3b-thinking-bf16", *VIEWS]
    for entry in entries["models"]:
        assert entry["kind"] == "model" and entry["status"] == "COMPLETE"
        assert isinstance(entry["local_subdir"], str) and entry["local_subdir"].startswith("models/")
        files = entry["source"]["files"]
        assert type(entry["files"]) is int and entry["files"] == len(files) > 0
        assert type(entry["size_bytes"]) is int and entry["size_bytes"] == sum(f["size_bytes"] for f in files) > 0
        assert len({f["filename"] for f in files}) == len(files)
        assert all(_HEX64.fullmatch(f["sha256"]) for f in files)
        assert entry["verification"]["payload_files"] == entry["files"]
        assert entry["verification"]["payload_size_bytes"] == entry["size_bytes"]
        assert set(entry["profiles"]) <= {"speech-aware-tools", *entries["profiles"]}


def test_the_port_fragment_names_the_bf16_entry_under_its_materialization_root(entries):
    fragment = json.loads(PORT_FRAGMENT.read_text(encoding="utf-8"))
    bf16 = entries["models"][0]
    assert fragment["model_lock_key"] == bf16["name"]
    # The lock stays root-relative; the fragment alone says where the root is.
    assert fragment["model_materialization_root"].startswith("/")
    assert not bf16["local_subdir"].startswith("/")
    assert bf16["source"]["kind"] == "hf" and re.fullmatch(r"[0-9a-f]{40}", bf16["revision"])
    assert bf16["revision"] == bf16["verification"]["resolved_revision"]
    assert [f["filename"] for f in bf16["source"]["files"]] == bf16["source"]["include"]
    shards = [f for f in bf16["source"]["files"] if f["filename"].endswith(".safetensors")]
    assert len(shards) == 16


def test_each_view_is_identified_by_its_receipt_hash(entries):
    for entry in entries["models"][1:]:
        assert entry["lifecycle"] == "DERIVED" and entry["source"]["kind"] == "derived"
        assert entry["local_subdir"] == f"models/serving-views/{entry['name']}"
        receipt = {f["filename"]: f for f in entry["source"]["files"]}["serving_view.json"]
        assert entry["revision"].endswith(receipt["sha256"])
        assert entry["verification"]["receipt_sha256"] == receipt["sha256"]
        assert "speech-aware-tools" in entry["profiles"]
        assert entry["source"]["args"][-1] == entry["name"]


def test_the_amendment_record_covers_exactly_the_new_entries(entries):
    amendment = entries["amendment"]
    assert amendment["date"] == "2026-10-28"
    registered = [name for change in amendment["changes"] if change["kind"] == "asset-registration"
                  for name in change["entries"]]
    assert sorted(registered) == sorted(entry["name"] for entry in entries["models"])


def test_the_amendment_refuses_any_lock_but_the_pristine_one(apply, entries):
    with pytest.raises(apply.AmendmentError, match="pristine"):
        apply.amend(b"{}\n", entries)


@pytest.mark.skipif(not LIVE_LOCK.is_file(), reason="umbrella lock not adjacent")
def test_an_amended_live_lock_resolves_every_new_entry_verbatim(apply, entries):
    raw = LIVE_LOCK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != apply.AMENDED_LOCK_SHA256:
        pytest.skip("the adjacent umbrella lock does not carry this amendment")
    lock = load_lock(LIVE_LOCK)
    for entry in entries["models"]:
        assert lock_entry(lock, entry["name"]) == entry
    assert lock["amendments"][0] == entries["amendment"]
