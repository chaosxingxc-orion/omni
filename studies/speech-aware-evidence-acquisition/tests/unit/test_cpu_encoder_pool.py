"""The port's CPU encoder runtime: batched seals equal single seals, and the pinned worker pool."""

import hashlib
import io
import os
import threading

import numpy as np
import pytest
import soundfile as sf

from speech_aware_evidence_acquisition.core import cpu_encoder_pool as pool_module
from speech_aware_evidence_acquisition.core import speech_preparation_runtime as runtime
from speech_aware_evidence_acquisition.core.cpu_encoder_pool import (
    CPU_INVENTORY_SCHEMA,
    CpuEncoderPool,
    CpuEncoderPoolError,
    CpuPassageEncoder,
    check_cpu_inventory,
    worker_slices,
)
from speech_aware_evidence_acquisition.core.cpu_encoder_worker import read_frame, write_frame
from speech_aware_evidence_acquisition.core.preparation import PreparationError
from speech_aware_evidence_acquisition.core.preparation_speech_queries import (
    prepare_speech_query_vectors,
    prepare_text_facet_vectors,
)

_SHA = "a" * 64


def _rows(tmp_path, *, like=None):
    """Synthetic rows; ``like`` copies another root's audio bytes (libsndfile stamps the time into FLOAT WAVs)."""
    rows = []
    for name, frames in (("one", 250), ("two", 20), ("three", 600)):
        path = tmp_path / f"{name}.wav"
        if like is None:
            sf.write(path, np.zeros(frames, dtype="float32"), 2, subtype="FLOAT")
        else:
            path.write_bytes((like / path.name).read_bytes())
        rows.append({
            "id": name, "media_relpath": path.name, "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_frames": frames, "source_sample_rate_hz": 2,
        })
    rows.append(dict(rows[1], id="alias"))  # an equal span under another sample identity
    return rows


def _vector(source):
    return [float(source["start_frame"] + 1), float(source["end_frame"])]


def _speech(root, rows, **kwargs):
    return prepare_speech_query_vectors(
        root / "prepared.json", rows=rows, data_root=root, cache_root=root / "vectors",
        encoder_identity={"id": "synthetic", "weights": _SHA}, protocol_id="p-v1", dimension=2, **kwargs)


def _tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json")) if "vectors" in path.parts or path.name == "prepared.json"
    }


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_batched_speech_seals_equal_the_single_contact_path(tmp_path, batch_size):
    single_root, batched_root = tmp_path / "single", tmp_path / "batched"
    for root in (single_root, batched_root):
        root.mkdir()
    single = _speech(single_root, _rows(single_root), encoder=_vector)
    windows = []
    batched = _speech(batched_root, _rows(batched_root, like=single_root), encoder=pytest.fail,
                      encode_batch=lambda sources: windows.append(len(sources)) or [_vector(s) for s in sources],
                      batch_size=batch_size)
    assert batched["artifact"]["sha256"] == single["artifact"]["sha256"]
    assert batched["accounting"] == single["accounting"]
    assert _tree(batched_root) == _tree(single_root)
    assert max(windows) <= batch_size and sum(windows) == single["accounting"]["encoder_calls"]


def test_batched_speech_counts_cache_hits_inside_a_window(tmp_path):
    rows = _rows(tmp_path)
    first = _speech(tmp_path, rows[:2], encoder=_vector)
    (tmp_path / "prepared.json").unlink()
    spent = []
    second = _speech(tmp_path, rows, encoder=pytest.fail,
                     encode_batch=lambda sources: spent.extend(sources) or [_vector(s) for s in sources],
                     batch_size=4)
    assert second["accounting"]["cache_hits"] == first["accounting"]["encoder_calls"]
    assert second["accounting"]["encoder_calls"] == len(spent)
    assert second["accounting"]["deduplicated_references"] == 1


def test_a_failed_window_keeps_its_start_markers_and_refuses_replay(tmp_path):
    rows = _rows(tmp_path)
    with pytest.raises(PreparationError, match="count differs"):
        _speech(tmp_path, rows, encoder=pytest.fail, encode_batch=lambda sources: [], batch_size=2)
    assert list((tmp_path / "vectors").glob("*.encoder-started.json"))
    with pytest.raises(PreparationError, match="new registered attempt"):
        _speech(tmp_path, rows, encoder=pytest.fail, encode_batch=lambda s: [_vector(x) for x in s], batch_size=2)


def test_batched_text_facets_equal_the_single_contact_path(tmp_path):
    facets = [{"id": f"facet:{index}", "text": f"text {index}"} for index in range(7)]

    def run(root, **kwargs):
        return prepare_text_facet_vectors(
            root / "set.json", facets=facets, cache_root=root / "cache", encoder_identity={"inventory_sha256": _SHA},
            snapshot_sha256=_SHA, protocol_id="b" * 64, dimension=2, **kwargs)

    single = run(tmp_path / "single", encoder=lambda text: [1.0, float(len(text))])
    batched = run(tmp_path / "batched", encoder=pytest.fail, batch_size=3,
                  encode_batch=lambda texts: [[1.0, float(len(text))] for text in texts])
    assert batched["artifact"]["sha256"] == single["artifact"]["sha256"]
    assert batched["accounting"] == single["accounting"]
    for name in ("single", "batched"):
        assert (tmp_path / name / "set.json").exists()
    assert sorted(p.read_bytes() for p in (tmp_path / "single/cache").glob("*.json")) == sorted(
        p.read_bytes() for p in (tmp_path / "batched/cache").glob("*.json"))


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_size_must_be_a_positive_integer(tmp_path, batch_size):
    with pytest.raises(PreparationError, match="batch_size"):
        _speech(tmp_path, _rows(tmp_path), encoder=_vector, encode_batch=lambda s: [], batch_size=batch_size)


def _inventory(**changes):
    first = min(os.sched_getaffinity(0))
    return {
        "schema": CPU_INVENTORY_SCHEMA, "model_dir_relpath": "models/x", "files": {},
        "packages": {"torch": "2.10.0", "transformers": "4.57.1"}, "runtime": "cpu-float32",
        "python_executable": "/usr/bin/python3", "workers": 2, "threads_per_worker": 1, "first_core": first,
        **changes,
    }


@pytest.mark.parametrize("changes", [
    {"runtime": "cuda-bfloat16"}, {"runtime": "cpu-int8"}, {"workers": 0}, {"threads_per_worker": True},
    {"first_core": -1}, {"python_executable": "python3"}, {"packages": {"torch": "2.10.0"}}, {"extra": 1},
])
def test_cpu_inventory_shape_refuses(changes):
    with pytest.raises(CpuEncoderPoolError):
        check_cpu_inventory(_inventory(**changes))


def test_worker_slices_are_disjoint_and_contiguous():
    assert worker_slices(_inventory(workers=3, threads_per_worker=4, first_core=128)) == [
        "128-131", "132-135", "136-139"]


class _FakeWorker:
    """A worker double speaking the real frame protocol over real pipes."""

    def __init__(self, command, *, report, delay=None, **_kwargs):
        request_read, request_write = os.pipe()
        reply_read, reply_write = os.pipe()
        self.stdin, self.stdout = os.fdopen(request_write, "wb"), os.fdopen(reply_read, "rb")
        self.command, self.cores = command, command[-3]
        inbound, outbound = os.fdopen(request_read, "rb"), os.fdopen(reply_write, "wb")

        def serve():
            write_frame(outbound, report)
            while (request := read_frame(inbound)) is not None:
                text = request["content"][0]["text"]
                if text.endswith("boom"):
                    write_frame(outbound, {"error": "ValueError: boom"})
                else:
                    write_frame(outbound, {"vector": [float(len(text)), float(int(self.cores.split("-")[0]))]})
            outbound.close()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()

    def wait(self, timeout=None):
        self.thread.join(timeout)

    def kill(self):
        pass


def _spawn(report_changes=None):
    def spawn(command, **kwargs):
        report = {"ready": True, "packages": dict(_inventory()["packages"]), "threads": 1}
        report.update(report_changes or {})
        return _FakeWorker(command, report=report, **kwargs)
    return spawn


def test_pool_keeps_caller_order_and_spreads_requests_over_workers():
    pool = CpuEncoderPool(_inventory(), "/models/x", spawn=_spawn())
    try:
        requests = [{"content": [{"type": "text", "text": "x" * (index + 1)}]} for index in range(5)]
        vectors = pool.map(requests)
    finally:
        pool.close()
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len({vector[1] for vector in vectors}) == 2


@pytest.mark.parametrize("report", [{"packages": {"torch": "2.9.0", "transformers": "4.57.1"}}, {"threads": 4},
                                    {"ready": False}])
def test_pool_refuses_a_worker_that_differs_from_its_pins(report):
    with pytest.raises(CpuEncoderPoolError):
        CpuEncoderPool(_inventory(), "/models/x", spawn=_spawn(report))


def test_pool_refuses_core_slices_outside_its_affinity():
    with pytest.raises(CpuEncoderPoolError, match="exceed"):
        CpuEncoderPool(_inventory(first_core=max(os.sched_getaffinity(0)) + 1), "/models/x", spawn=_spawn())


def test_worker_failure_surfaces_as_an_error():
    pool = CpuEncoderPool(_inventory(), "/models/x", spawn=_spawn())
    try:
        with pytest.raises(CpuEncoderPoolError, match="boom"):
            pool.map([{"content": [{"type": "text", "text": "boom"}]}])
    finally:
        pool.close()


def test_passage_backend_starts_its_pool_lazily_and_reports_concurrency():
    events, made = [], []

    class Pool:
        def __init__(self, inventory, model_dir):
            made.append(model_dir)

        def map(self, requests):
            return [[1.0, float(len(r["content"][0]["text"]))] for r in requests]

        def close(self):
            events.append(("closed",))

    backend = CpuPassageEncoder(_inventory(workers=3), {"passage_prompt": "passage: ", "max_sequence_tokens": 8192},
                                None, lambda phase, **facts: events.append((phase,)), model_dir="/m", pool_cls=Pool)
    assert backend.concurrency == 3 and not made
    assert backend.decode_batch(["ab", "abcd"]) == [[1.0, 11.0], [1.0, 13.0]]
    assert backend("a") == [1.0, 10.0]
    backend.close()
    assert made == ["/m"]
    assert [e[0] for e in events].count("model_load_started") == 1 and events[-1] == ("closed",)


def test_cpu_inventory_runs_only_the_default_cpu_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_verify_encoder_files", lambda inventory, data_root: None)
    chosen = runtime.verified_backend(_inventory(), tmp_path, runtime.NemotronPassageEncoder,
                                      default=runtime.NemotronPassageEncoder, served_cls=object,
                                      cpu_cls=runtime._cpu_passage_backend)
    assert chosen is runtime._cpu_passage_backend
    with pytest.raises(runtime.SpeechPreparationError, match="CPU worker pool"):
        runtime.verified_backend(_inventory(), tmp_path, lambda *a: None, default=runtime.NemotronPassageEncoder,
                                 served_cls=object, cpu_cls=runtime._cpu_passage_backend)


def test_only_a_concurrent_backend_receives_batch_contacts():
    calls = []

    def contact(sources, *, batch=False):
        calls.append((tuple(sources), batch))
        return ["v"] * len(sources)

    class Concurrent:
        concurrency = 4

    assert runtime._batch_contact(object(), contact) == {}
    assert runtime._batch_contact(type("One", (), {"concurrency": 1})(), contact) == {}
    wiring = runtime._batch_contact(Concurrent(), contact)
    assert wiring["batch_size"] == 4
    assert wiring["encode_batch"](["a", "b"]) == ["v", "v"] and calls == [(("a", "b"), True)]


def test_frames_round_trip():
    stream = io.BytesIO()
    write_frame(stream, {"vector": [0.5, 1.0]})
    stream.seek(0)
    assert read_frame(stream) == {"vector": [0.5, 1.0]}
    assert read_frame(stream) is None
    assert pool_module.CPU_INVENTORY_SCHEMA == "saea-cpu-encoder-inventory-v1"


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8"])
def test_waves_cross_interpreters_as_builtins_and_come_back_exactly(dtype):
    from speech_aware_evidence_acquisition.core.cpu_encoder_worker import pack_wave, unpack_wave

    wave = np.random.default_rng(1).standard_normal(97).astype(dtype)
    packed = pack_wave(wave)
    assert set(packed) == {"dtype", "data"} and type(packed["data"]) is bytes and type(packed["dtype"]) is str
    restored = unpack_wave(packed)
    assert restored.dtype == wave.dtype.newbyteorder("<") and np.array_equal(restored, wave)
    with pytest.raises(ValueError):
        pack_wave(np.zeros((2, 2), dtype="float32"))
