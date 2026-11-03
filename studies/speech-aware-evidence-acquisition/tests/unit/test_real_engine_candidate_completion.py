import importlib.util
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location('candidate_runner', Path(__file__).resolve().parents[2] / 'scripts/analysis/run_real_engine_candidate.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_generation_completion_does_not_hide_failed_score():
    result = {'pending': [], 'completed': ['r'], 'states': {'r': {'state': 'failed', 'reason': 'official scoring failed'}}}
    with pytest.raises(RuntimeError, match='finalization failed'):
        m.assert_campaign_success(result, 'r')
    result['states']['r']['state'] = 'scored'
    m.assert_campaign_success(result, 'r')


def test_release_pins_immutable_inputs_and_rejects_corruption(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
    receipts = FlightReceipts('r', 'a')
    seal = {}
    for key in ('generation_manifest', 'score_ledger_cells', 'score_jobs'):
        path = tmp_path / key
        path.write_bytes(b'sealed')
        seal[key + '_path'] = str(path)
        seal[key + '_sha256'] = hashlib.sha256(b'sealed').hexdigest()
    receipts.record('seal', seal)
    runtime = SimpleNamespace(run_id='r', attempt_id='a', receipts=receipts)
    spec = {'run_id': 'r', 'attempt_id': 'a', 'profile': 'p', 'service_session': 's'}
    def release():
        return m.generation_release_document(runtime, candidate='c', spec=spec,
                                             commit='source', rows=500, elapsed=12)
    doc = release()
    assert doc['generation_lease_released'] and not doc['scoring_complete']
    assert doc['consumer_requires_own_lease_and_readiness']
    assert doc['rows'] == 500 and len(doc['artifacts']) == 3
    assert doc['seal_receipt'] == receipts.require('seal')
    runtime.attempt_id = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        release()
    runtime.attempt_id = 'a'
    (tmp_path / 'score_jobs').write_bytes(b'changed')
    with pytest.raises(ValueError, match='sealed input changed'):
        release()
    (tmp_path / 'score_jobs').write_bytes(b'sealed')
    receipts.record('score', {})
    with pytest.raises(ValueError, match='precede scoring'):
        release()


@pytest.mark.parametrize('mode', ['normal', 'debug'])
def test_candidate_stages_preserve_calls_and_record_failure(tmp_path, monkeypatch, mode):
    import json
    import os
    from speech_aware_evidence_acquisition.core import stage_timing
    sink = tmp_path/'stages.jsonl'
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH', str(sink))
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE', mode)
    monkeypatch.delenv('SAEA_OBSERVABILITY_CONFIG', raising=False)
    monkeypatch.delenv('SAEA_OBSERVABILITY_RUN_ID', raising=False)
    monkeypatch.delenv('SAEA_OBSERVABILITY_ATTEMPT_ID', raising=False)
    monkeypatch.setattr(stage_timing, '_fd', None)
    monkeypatch.setattr(stage_timing, '_path', None)
    spec = {'run_id': 'run', 'attempt_id': 'attempt'}
    value = object()
    calls = []
    def operation(arg, *, option):
        calls.append((arg, option))
        return value
    error = ValueError('private failure payload')
    def broken(arg):
        assert arg is value
        raise error
    try:
        for stage in ('runtime_build', 'prepared_artifact_verification', 'campaign'):
            assert m.measured_stage(stage, spec, operation, value, option=7) is value
        with pytest.raises(ValueError) as caught:
            m.measured_stage('prepared_artifact_verification', spec, broken, value)
        assert caught.value is error
        assert calls == [(value, 7)] * 3
        events = [json.loads(line) for line in sink.read_text().splitlines()]
        assert [e['stage'] for e in events] == [
            'candidate_runtime_build', 'candidate_prepared_artifact_verification',
            'candidate_campaign', 'candidate_prepared_artifact_verification']
        assert [e['status'] for e in events] == ['ok', 'ok', 'ok', 'error']
        assert all(e['run_id'] == 'run' and e['attempt_id'] == 'attempt'
                   and e['observability_mode'] == mode and e['elapsed_seconds'] >= 0
                   for e in events)
        assert 'private failure payload' not in sink.read_text()
    finally:
        if stage_timing._fd is not None:
            os.close(stage_timing._fd)


def test_candidate_timing_disabled_does_not_require_identity(monkeypatch):
    monkeypatch.delenv('SAEA_STAGE_TIMING_PATH', raising=False)
    value = object()
    assert m.measured_stage('runtime_build', {}, lambda arg: arg, value) is value
