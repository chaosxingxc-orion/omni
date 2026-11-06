"""The real flight warms its own transport cache before inference, including resume dispatch."""
import pytest
from speech_aware_evidence_acquisition.core.model import transport
from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntimeError
from test_flight_runtime import _make_runtime


@pytest.fixture
def cache(monkeypatch):
    transport._clear_audio_cache()
    monkeypatch.setattr(transport, '_AUDIO_CACHE_ENABLED', True)
    monkeypatch.setattr(transport, '_AUDIO_CACHE_MAX_ENTRIES', 32)
    monkeypatch.setattr(transport, '_AUDIO_CACHE_MAX_BYTES', 4096)
    monkeypatch.setenv('SAEA_AUDIO_PRELOAD_WORKERS', '4')
    yield
    transport._clear_audio_cache()


def test_flight_preload_populates_same_cache_without_changing_config(tmp_path, monkeypatch, cache):
    runtime = _make_runtime(tmp_path, happy_path=True)
    runtime.preflight()
    before = runtime._composed.config_hash
    for sample in runtime._samples:
        (runtime._data_root / sample.media_relpath).write_bytes(b'frozen audio')
    receipt = runtime.preprocess()
    assert receipt['audio_cache']['loaded'] == len(runtime._samples)
    assert runtime._composed.config_hash == before
    # A transport cache hit performs stat validation but does not reopen media.
    from pathlib import Path
    original = Path.read_bytes
    media = {runtime._data_root / sample.media_relpath for sample in runtime._samples}
    def read(path):
        assert path not in media, 'warm cache must serve inference without rereading media'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    for sample in runtime._samples:
        transport._load_audio_part(runtime._data_root, sample.media_relpath, 'test')


def test_missing_media_fails_before_promoter_or_model(tmp_path, cache):
    runtime = _make_runtime(tmp_path, happy_path=True)
    runtime.preflight()
    runtime._packet_promoter = lambda **kw: pytest.fail('must reject missing media before promotion')
    with pytest.raises(FlightRuntimeError, match='preload failed before model contact'):
        runtime.preprocess()


@pytest.mark.parametrize('raw', ['', '0', '65', '-1', 'true', '2.5', ' 4', '４'])
def test_invalid_preload_workers_fail_closed(monkeypatch, cache, raw):
    monkeypatch.setenv('SAEA_AUDIO_PRELOAD_WORKERS', raw)
    with pytest.raises(Exception, match='integer'):
        transport.environment_audio_preload_workers(has_runtime_policy=False)


def test_conflicting_policy_and_disabled_cache_are_rejected(monkeypatch, cache):
    with pytest.raises(Exception, match='conflicts'):
        transport.environment_audio_preload_workers(has_runtime_policy=True)
    monkeypatch.setattr(transport, '_AUDIO_CACHE_ENABLED', False)
    with pytest.raises(Exception, match='positive capacity'):
        transport.environment_audio_preload_workers(has_runtime_policy=False)
    monkeypatch.delenv('SAEA_AUDIO_PRELOAD_WORKERS')
    assert transport.environment_audio_preload_workers(has_runtime_policy=False) is None
