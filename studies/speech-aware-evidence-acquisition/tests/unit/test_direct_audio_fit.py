"""Direct FIT measures original raw FLAC/WAV wire bytes before any contact."""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from speech_aware_evidence_acquisition.core.model.transport import _clear_audio_cache, _load_audio_part
from speech_aware_evidence_acquisition.core.preparation import PreparationError
from speech_aware_evidence_acquisition.core.preparation_fit import rendered_wav_seconds
from test_direct_only_fit import _Runtime, _build, _plan, _request, ROWS


def _audio(format_name, subtype, *, frames=24000):
    stream = io.BytesIO()
    soundfile.write(stream, np.zeros((frames, 1)), 16000, format=format_name, subtype=subtype)
    return stream.getvalue()


def _body(raw, declared):
    return {"messages": [{"role": "user", "content": [{"type": "input_audio",
        "input_audio": {"data": base64.b64encode(raw).decode(), "format": declared}}]}]}


@pytest.fixture
def audio_cache(monkeypatch):
    import speech_aware_evidence_acquisition.core.model.transport as module
    _clear_audio_cache()
    monkeypatch.setattr(module, "_AUDIO_CACHE_ENABLED", True)
    monkeypatch.setattr(module, "_AUDIO_CACHE_MAX_ENTRIES", 4)
    monkeypatch.setattr(module, "_AUDIO_CACHE_MAX_BYTES", 1024)
    yield module
    _clear_audio_cache()


def test_transport_audio_cache_reuses_bytes_and_invalidates_on_file_change(tmp_path, monkeypatch, audio_cache):
    path = tmp_path / "query.wav"
    path.write_bytes(b"first-audio")
    reads = []
    original = Path.read_bytes

    def counted(self):
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counted)
    _, first = _load_audio_part(tmp_path, "query.wav", "query")
    _, second = _load_audio_part(tmp_path, "query.wav", "query")
    assert first == second and len(reads) == 1
    path.write_bytes(b"second-audio")
    _, changed = _load_audio_part(tmp_path, "query.wav", "query")
    assert changed != first and len(reads) == 2


@pytest.mark.parametrize("limit", ["bytes", "entries", "oversized", "disabled"])
def test_transport_audio_cache_bounds(tmp_path, monkeypatch, audio_cache, limit):
    if limit == "bytes":
        monkeypatch.setattr(audio_cache, "_AUDIO_CACHE_MAX_BYTES", 8)
    elif limit == "entries":
        monkeypatch.setattr(audio_cache, "_AUDIO_CACHE_MAX_ENTRIES", 1)
    elif limit == "oversized":
        monkeypatch.setattr(audio_cache, "_AUDIO_CACHE_MAX_BYTES", 4)
    else:
        monkeypatch.setattr(audio_cache, "_AUDIO_CACHE_ENABLED", False)
    for name, raw in (("a.wav", b"123456"), ("b.wav", b"abcdef")):
        (tmp_path / name).write_bytes(raw)
        _, part = _load_audio_part(tmp_path, name, "query")
        assert base64.b64decode(part["input_audio"]["data"]) == raw
    expected = 0 if limit in ("oversized", "disabled") else 1
    assert len(audio_cache._AUDIO_CACHE) == expected
    assert audio_cache._AUDIO_CACHE_BYTES == expected * 8


def test_concurrent_audio_cache_misses_charge_bytes_once(tmp_path, monkeypatch, audio_cache):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    (tmp_path / "q.wav").write_bytes(b"123456")
    barrier = Barrier(4)
    original = Path.read_bytes

    def read(self):
        raw = original(self)
        barrier.wait(timeout=5)
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _load_audio_part(tmp_path, "q.wav", "query"), range(4)))
    assert all(result == results[0] for result in results)
    assert len(audio_cache._AUDIO_CACHE) == 1
    assert audio_cache._AUDIO_CACHE_BYTES == 8


def test_audio_mutation_during_read_is_rejected(tmp_path, monkeypatch, audio_cache):
    from speech_aware_evidence_acquisition.core.model import FrozenCoreError
    path = tmp_path / "q.wav"
    path.write_bytes(b"123456")
    original = Path.read_bytes

    def read(self):
        raw = original(self)
        self.write_bytes(b"changed-size")
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(FrozenCoreError, match="media changed"):
        _load_audio_part(tmp_path, "q.wav", "query")
    assert not audio_cache._AUDIO_CACHE and audio_cache._AUDIO_CACHE_BYTES == 0


@pytest.mark.parametrize("format_name,subtype,declared", [
    ("WAV", "PCM_16", "wav"), ("WAV", "FLOAT", "wav"),
    ("WAVEX", "PCM_24", "wav"), ("FLAC", "PCM_16", "flac"),
])
def test_actual_audio_header_duration_and_immutable_wire(format_name, subtype, declared):
    body = _body(_audio(format_name, subtype), declared)
    before = json.dumps(body, sort_keys=True)
    assert rendered_wav_seconds(body) == 1.5
    assert json.dumps(body, sort_keys=True) == before


@pytest.mark.parametrize("declared", ["wav", "mp3", "ogg", None, "FLAC"])
def test_flac_never_accepts_mislabeled_or_unsupported_format(declared):
    with pytest.raises(PreparationError, match="format"):
        rendered_wav_seconds(_body(_audio("FLAC", "PCM_16"), declared))


@pytest.mark.parametrize("raw,declared", [
    (b"fLaC", "flac"), (b"RIFF0000WAVE", "wav"), (b"not audio", "wav"),
])
def test_malformed_audio_has_no_duration_fallback(raw, declared):
    body = _body(raw, declared)
    body["audio_seconds"] = 1.5
    with pytest.raises(PreparationError, match="header"):
        rendered_wav_seconds(body)


def test_empty_audio_cannot_claim_zero_cost():
    with pytest.raises(PreparationError, match="invalid frame"):
        rendered_wav_seconds(_body(_audio("WAV", "PCM_16", frames=0), "wav"))


@pytest.mark.parametrize("change", [
    {"frames": 0}, {"samplerate": 0}, {"channels": 0}, {"format": "OGG"},
])
def test_inconsistent_reader_metadata_is_refused(monkeypatch, change):
    raw = _audio("FLAC", "PCM_16")
    facts = {"frames": 24000, "samplerate": 16000, "channels": 1, "format": "FLAC", **change}
    monkeypatch.setattr(soundfile, "info", lambda stream: SimpleNamespace(**facts))
    with pytest.raises(PreparationError):
        rendered_wav_seconds(_body(raw, "flac"))


def _raw_flac_gate(tmp_path, *, cap, changed_digest=False):
    runtime = _Runtime(tmp_path)
    requests = []
    for i, row in enumerate(ROWS):
        path = tmp_path / f"media/{row}.flac"
        path.parent.mkdir(exist_ok=True)
        soundfile.write(path, np.full((24000, 1), i / 16), 16000, format="FLAC", subtype="PCM_16")
        rel = str(path.relative_to(tmp_path))
        requests.append(_request(row, media_relpath=rel, audio_seconds=0.001,
            raw_speech_query={"media_relpath": rel, "sha256":
                "f" * 64 if changed_digest and i == 0 else hashlib.sha256(path.read_bytes()).hexdigest()}))

    class RawTransport:
        def render_request(self, raw):
            payload = json.loads(raw)
            _, part = _load_audio_part(tmp_path, payload["media_relpath"], "query")
            return {"model": "model", "max_tokens": 4,
                "messages": [{"role": "user", "content": [part]}]}

    runtime._transport = RawTransport()
    gate, tokenizer = _build(runtime, plan=_plan(planned_speech_audio_seconds=cap))
    return gate, tokenizer, tuple(requests)


def test_real_raw_flac_wire_passes_whole_wave_fit_at_exact_ceiling(tmp_path):
    gate, tokenizer, batch = _raw_flac_gate(tmp_path, cap=6)
    gate("direct", batch)
    assert len(tokenizer.calls) == 4
    assert all(call[2]["messages"][0]["content"][0]["input_audio"]["format"] == "flac"
        for call in tokenizer.calls)


def test_real_raw_flac_uses_header_not_payload_duration_before_tokenizer(tmp_path):
    gate, tokenizer, batch = _raw_flac_gate(tmp_path, cap=5)
    with pytest.raises(PreparationError, match="audio ceiling: 6 > 5"):
        gate("direct", batch)
    assert not tokenizer.calls and not list(tmp_path.rglob("*.contacts.jsonl"))


def test_real_raw_flac_still_requires_exact_source_sha_before_tokenizer(tmp_path):
    gate, tokenizer, batch = _raw_flac_gate(tmp_path, cap=6, changed_digest=True)
    with pytest.raises(PreparationError, match="rendered audio differs"):
        gate("direct", batch)
    assert not tokenizer.calls
