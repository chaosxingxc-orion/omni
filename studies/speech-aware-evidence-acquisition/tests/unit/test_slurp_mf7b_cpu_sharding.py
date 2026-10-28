"""MF-7b key pass on the CPU runtime: sharding, merge, receipt runtime fields and the kus3 facets-only mode."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import _fake_mf7b_backend as fake
from speech_aware_evidence_acquisition.analysis import slurp_mf7b_sharding as sharding
from speech_aware_evidence_acquisition.analysis.slurp_mf7b_vectors import write_vectors_jsonl

REPO = Path(__file__).resolve().parents[2]
FAKE = "_fake_mf7b_backend"


def _items(count):
    return [(f"facet:content:{index}", f"utterance number {index} " + "x" * (index % 5)) for index in range(count)]


def _single_process(items, prompt, batch, runtime="cpu-float32"):
    device = runtime.split("-")[0]
    processor, model, template = fake.load_stack("m", runtime=runtime)
    ids, vectors = [], []
    for offset in range(0, len(items), batch):
        chunk = items[offset : offset + batch]
        rendered = [fake.render(processor, template, [{"type": "text", "text": prompt + t}]) for _i, t in chunk]
        vectors.extend(fake.encode(processor, model, rendered, None, device=device))
        ids.extend(i for i, _t in chunk)
    return ids, vectors


def _in_process(specs):
    return [(spec.worker, "ok", *sharding.encode_shard(spec)) for spec in specs]


def _cores(count):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < count:
        pytest.skip(f"needs {count} schedulable cores")
    return available[0], available[0] + count - 1


# ---- attention geometry -----------------------------------------------------------------------------------


def test_forced_geometry_reaches_every_shard_and_is_verified():
    items = _items(40)
    _ids, _vectors, _block, _rendered = sharding.sharded_text_pass(
        items, "passage: ", model_dir=Path("m"), runtime="cpu-float32", device="cpu", batch=16, workers=3,
        cores=_cores(3), attention_geometry="causal", backend=FAKE, launcher=_in_process)
    specs = []
    sharding.sharded_text_pass(
        items, "passage: ", model_dir=Path("m"), runtime="cpu-float32", device="cpu", batch=16, workers=3,
        cores=_cores(3), attention_geometry="causal", backend=FAKE, launcher=lambda s: specs.extend(s) or _in_process(s))
    assert {spec.attention_geometry for spec in specs} == {"causal"}
    with pytest.raises(sharding.ShardingError, match="measured attention geometry 'bidirectional'"):
        sharding.sharded_text_pass(
            items, "passage: ", model_dir=Path("m-mismeasured"), runtime="cpu-float32", device="cpu", batch=16,
            workers=3, cores=_cores(3), attention_geometry="causal", backend=FAKE, launcher=_in_process)


def test_as_shipped_geometry_is_never_measured():
    _ids, _vectors, _block, _rendered = sharding.sharded_text_pass(
        _items(20), "passage: ", model_dir=Path("m-mismeasured"), runtime="cpu-float32", device="cpu", batch=16,
        workers=2, cores=_cores(2), backend=FAKE, launcher=_in_process)


# ---- partition rules --------------------------------------------------------------------------------------


def test_chunks_are_the_single_process_batches():
    items = _items(37)
    chunks = sharding.chunk_items(items, 16)
    assert [len(c) for c in chunks] == [16, 16, 5]
    assert [x for c in chunks for x in c] == items


@pytest.mark.parametrize("count,workers", [(720, 4), (720, 7), (3, 3), (10, 1)])
def test_partition_is_contiguous_balanced_and_complete(count, workers):
    parts = sharding.partition_chunks(count, workers)
    assert [i for part in parts for i in part] == list(range(count))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_partition_and_slices_refuse_idle_workers():
    with pytest.raises(sharding.ShardingError):
        sharding.partition_chunks(3, 4)
    with pytest.raises(sharding.ShardingError):
        sharding.core_slices(10, 12, 4)
    assert sharding.core_slices(128, 191, 4) == [(128, 143), (144, 159), (160, 175), (176, 191)]
    assert sharding.core_slices(0, 6, 3) == [(0, 2), (3, 4), (5, 6)]


@pytest.mark.parametrize("text", ["128", "a-b", "9-3", "-1-4"])
def test_parse_cores_refuses_malformed_ranges(text):
    with pytest.raises(sharding.ShardingError):
        sharding.parse_cores(text)


# ---- merge ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 3, 5])
def test_sharded_pass_equals_single_process_bytes(tmp_path, workers):
    items = _items(77)
    reference = _single_process(items, "passage: ", 16)
    ids, vectors, block, rendered = sharding.sharded_text_pass(
        items, "passage: ", model_dir=tmp_path, runtime="cpu-float32", device="cpu", batch=16,
        workers=workers, cores=(0, 9), backend=FAKE, launcher=_in_process)
    assert (ids, vectors) == reference
    assert rendered == "passage: " + items[0][1]
    assert block["forwards"] == block["chunks"] == 5 and block["workers"] == workers
    a = write_vectors_jsonl(tmp_path / "a.jsonl", *reference).read_bytes()
    b = write_vectors_jsonl(tmp_path / "b.jsonl", ids, vectors).read_bytes()
    assert a == b


def test_merge_refuses_failed_missing_and_duplicate_chunks():
    good = [(0, "ok", [(0, ["a"], [[1.0]]), (1, ["b"], [[2.0]])], "r")]
    assert sharding.merge_shards(good, 2) == (["a", "b"], [(1.0,), (2.0,)])
    with pytest.raises(sharding.ShardingError, match="failed"):
        sharding.merge_shards(good + [(1, "error", "RuntimeError: x", None)], 2)
    with pytest.raises(sharding.ShardingError, match="of 3 chunks"):
        sharding.merge_shards(good, 3)
    with pytest.raises(sharding.ShardingError, match="twice"):
        sharding.merge_shards(good + [(1, "ok", [(1, ["b"], [[2.0]])], None)], 2)


def test_spawned_workers_are_pinned_and_merge_identically(tmp_path, monkeypatch):
    items = _items(50)
    reference = _single_process(items, "passage: ", 16)
    monkeypatch.setenv("_FAKE_EXPECT_PINNED", "1")  # inherited by the spawned workers only
    ids, vectors, block, _ = sharding.sharded_text_pass(
        items, "passage: ", model_dir=tmp_path, runtime="cpu-float32", device="cpu", batch=16,
        workers=2, cores=_cores(4), backend=FAKE)
    assert (ids, vectors) == reference
    assert len(block["core_slices"]) == 2


def test_a_failing_spawned_worker_aborts_the_pass(tmp_path):
    items = _items(40) + [("facet:content:boom", "boom")]
    with pytest.raises(sharding.ShardingError, match="fake encoder failure"):
        sharding.sharded_text_pass(
            items, "passage: ", model_dir=tmp_path, runtime="cpu-float32", device="cpu", batch=16,
            workers=3, cores=_cores(3), backend=FAKE)


# ---- the MF-7b pass script --------------------------------------------------------------------------------


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mf7b(tmp_path, monkeypatch):
    module = _load_script("slurp_mf7b_nemotron_embedding_pass")
    model_dir = tmp_path / module.MODEL_SUBDIR
    (model_dir / "additional_chat_templates").mkdir(parents=True)
    for name in ("config.json", "modeling_nv_omni_embed.py", "tokenizer.json",
                 "additional_chat_templates/sentence_transformers.jinja",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (model_dir / name).write_text(name)
    calls = {"load": [], "encode": []}

    def load_stack(path, **kwargs):
        calls["load"].append(kwargs)
        return fake.load_stack(path, **kwargs)

    def encode(*args, **kwargs):
        calls["encode"].append(kwargs.get("device"))
        return fake.encode(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a keys-only pass must not read DEV rows or drafts")

    items = _items(41)
    monkeypatch.setattr(module, "data_root", lambda: tmp_path)
    monkeypatch.setattr(module, "load_stack", load_stack)
    monkeypatch.setattr(module, "encode", encode)
    monkeypatch.setattr(module, "render", fake.render)
    monkeypatch.setattr(module, "dev_rows", forbidden)
    monkeypatch.setattr(module, "draft_texts", forbidden)
    monkeypatch.setattr(module, "train_facets", lambda _root: [{"facet_id": i, "text": t} for i, t in items])
    monkeypatch.setattr(module, "label_key_texts", lambda _root: _items(5))
    monkeypatch.setattr(module, "sharded_text_pass",
                        functools.partial(sharding.sharded_text_pass, backend=FAKE, launcher=_in_process))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fake-torch"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="fake-transformers"))
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["mf7b", *argv])
        assert module.main() == 0
        out = tmp_path / "vectors/slurp-mf7b"
        return (json.loads((out / "nemotron.receipt.json").read_text()),
                (out / "nemotron-key-vectors.jsonl").read_bytes())

    return SimpleNamespace(run=run, calls=calls, items=items, root=tmp_path)


def test_default_path_is_the_historical_cuda_bfloat16_recipe(mf7b):
    receipt, data = mf7b.run("--sets", "keys")
    assert mf7b.calls["load"] == [{"runtime": "cuda-bfloat16", "attention_geometry": None}]
    assert set(mf7b.calls["encode"]) == {"cuda"}
    embedder = receipt["embedder"]
    assert (embedder["device"], embedder["dtype"], embedder["gpu"]) == ("cuda", "bfloat16", True)
    assert embedder["attention_geometry"] == "as-shipped"
    assert "dtype=bfloat16" in embedder["loader"]
    assert "gpu_hours" in receipt["accounting"] and "cpu_wall_hours" not in receipt["accounting"]
    block = receipt["sets"]["nemotron-train-text"]
    assert block["batch_size"] == 16 and "sharding" not in block
    assert receipt["accounting"]["forward_passes"] == 3 and receipt["accounting"]["embedding_tool_contacts"] == 41
    assert data == write_vectors_jsonl(mf7b.root / "ref.jsonl", *_single_process(
        mf7b.items, "passage: ", 16, runtime="cuda-bfloat16")).read_bytes()


def test_sharded_cpu_pass_writes_the_unsharded_bytes_and_records_its_runtime(mf7b, tmp_path):
    receipt, data = mf7b.run("--sets", "keys", "--runtime", "cpu-float32", "--workers", "3", "--cores", "0-5")
    assert mf7b.calls["load"] == []  # the parent loads no stack; each worker loads its own
    assert data == write_vectors_jsonl(tmp_path / "ref.jsonl", *_single_process(
        mf7b.items, "passage: ", 16)).read_bytes()
    embedder = receipt["embedder"]
    assert (embedder["device"], embedder["dtype"], embedder["gpu"]) == ("cpu", "float32", False)
    accounting = receipt["accounting"]
    assert accounting["gpu_hours"] == 0.0 and accounting["cpu_wall_hours"] >= 0.0
    assert accounting["forward_passes"] == 3 and accounting["embedding_tool_contacts"] == 41
    block = receipt["sets"]["nemotron-train-text"]
    assert block["batch_size"] == 16
    assert block["sharding"]["rule"] == sharding.PARTITION_RULE
    assert block["sharding"]["core_slices"] == ["0-1", "2-3", "4-5"]


def test_forced_geometry_is_recorded_and_a_resumed_pass_must_keep_it(mf7b):
    receipt, _data = mf7b.run("--sets", "keys", "--runtime", "cpu-float32", "--workers", "2", "--cores", "0-3",
                              "--attention-geometry", "causal")
    assert receipt["embedder"]["attention_geometry"] == "causal"
    with pytest.raises(SystemExit, match="same geometry"):
        mf7b.run("--sets", "labels", "--runtime", "cpu-float32")


def test_resumed_pass_merges_sets_and_refuses_a_runtime_switch(mf7b):
    mf7b.run("--sets", "keys", "--runtime", "cpu-float32", "--workers", "2", "--cores", "0-3")
    receipt, _ = mf7b.run("--sets", "labels", "--runtime", "cpu-float32")
    assert set(receipt["sets"]) == {"nemotron-train-text", "nemotron-label-text"}
    assert len(receipt["passes"]) == 2 and receipt["accounting"]["embedding_tool_contacts"] == 46
    with pytest.raises(SystemExit, match="same runtime"):
        mf7b.run("--sets", "labels")


@pytest.mark.parametrize("argv", [["--workers", "2", "--cores", "0-3"],
                                  ["--runtime", "cpu-float32", "--workers", "2"]])
def test_sharding_needs_the_cpu_runtime_and_a_core_range(mf7b, argv):
    with pytest.raises(SystemExit):
        mf7b.run("--sets", "keys", *argv)


# ---- kus3 facets-only conversion --------------------------------------------------------------------------


@pytest.fixture
def kus3(tmp_path, monkeypatch):
    module = _load_script("build_slurp_kus3_vectors")
    source = tmp_path / module.MF7B_DIR_RELPATH
    source.mkdir(parents=True)
    sets = {}
    for name, filename, rows in (
        ("nemotron-train-text", "nemotron-key-vectors.jsonl", _items(6)),
        ("q-speech-nemotron", "nemotron-q-speech-query-vectors.jsonl", [(f"s{i}", "") for i in range(4)]),
        ("q-speech-nemotron-fused", "nemotron-fused-q-speech-query-vectors.jsonl", [(f"s{i}", "") for i in range(4)]),
    ):
        ids = [i for i, _t in rows]
        vectors = [(0.5, 0.5, 0.5, 0.5 + index / 100.0) for index in range(len(ids))]
        path = write_vectors_jsonl(source / filename, ids, vectors)
        sets[name] = {"file": filename, "count": len(ids), "sha256": module._sha256(path)}
    (source / module.MF7B_RECEIPT_NAME).write_text(json.dumps({
        "exposure_ledger_row": module.MF7B_LEDGER_ROW, "sets": sets,
        "embedder": {"encoder_id": "e", "dims": 4, "pooling": "mean", "prompts": {}}}))
    monkeypatch.setattr(module, "data_root", lambda: tmp_path)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["kus3", *argv])
        return module.main()

    return SimpleNamespace(module=module, run=run, root=tmp_path)


def test_facets_only_writes_the_full_runs_facet_bytes_without_dev_material(kus3, tmp_path):
    # No sealed closed-list outputs exist: a facets-only run must not need them.
    assert kus3.run("--facets-only", "--receipt-dir", str(tmp_path / "narrow")) == 0
    out = kus3.root / kus3.module.OUT_DIR_RELPATH
    narrow = (out / "facet-vectors.jsonl").read_bytes()
    assert sorted(p.name for p in out.iterdir()) == ["facet-vectors.jsonl"]
    receipt = json.loads((tmp_path / "narrow/vectors.json").read_text())
    assert list(receipt["sets"]) == ["facet-vectors.jsonl"] and receipt["narrowed_to"]["sets"] == ["facet-vectors.jsonl"]

    closed = kus3.root / kus3.module.SEALED_CLOSED_LIST_OUTPUTS_RELPATH
    closed.parent.mkdir(parents=True)
    closed.write_text("".join(json.dumps({"sample_id": f"s{i}", "text": json.dumps({"scenario": "alarm"})}) + "\n"
                              for i in range(4)))
    assert kus3.run("--receipt-dir", str(tmp_path / "full")) == 0
    assert (out / "facet-vectors.jsonl").read_bytes() == narrow
    full = json.loads((tmp_path / "full/vectors.json").read_text())
    assert "narrowed_to" not in full and len(full["sets"]) == 4


def test_facets_only_refuses_the_historical_receipt_dir(kus3):
    with pytest.raises(SystemExit):
        kus3.run("--facets-only")
