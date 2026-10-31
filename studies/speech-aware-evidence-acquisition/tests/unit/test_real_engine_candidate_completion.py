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
