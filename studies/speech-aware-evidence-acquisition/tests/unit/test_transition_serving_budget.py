"""Real resident registration on fake /proc; no SSH, engines or model calls."""
import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

from _vllm_world import VllmWorld, PROTOCOL_ARGS
from speech_aware_evidence_acquisition import contracts
from speech_aware_evidence_acquisition.core.session import vllm_identity as vi
from speech_aware_evidence_acquisition.core.session import vllm_resident as vr

SPEC = importlib.util.spec_from_file_location('transition_serving_budget',
    Path(__file__).parents[2]/'scripts/environment/transition_serving_budget.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def write_pin(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.raw(value))
    return {'path': str(path), 'sha256': m.digest(path.read_bytes())}


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = VllmWorld(tmp_path)
    w.spawn('tp2-a', pid=4200001); w.spawn('tp2-b', pid=4200002)
    repo = tmp_path/'repo'; data = w.materialization_root
    receipt = repo/'docs/receipts/runtime.json'; write_pin(receipt, w.receipt())
    source = {'transport': 'vllm', 'server_args': PROTOCOL_ARGS,
        'model_materialization_root': str(data), 'runtime_receipt': 'receipts/runtime.json',
        'decoding_params': {'max_tokens': 4096, 'temperature': 0},
        'reasoning_budget_tokens': 3072, 'answer_reserve_tokens': 1024}
    target = copy.deepcopy(source); target['decoding_params']['max_tokens'] = 16384
    target['answer_reserve_tokens'] = 13312
    observations = {name: {'pid': pid, 'started_at': '2026-10-30T00:00:00Z',
        'startup_stdout_sha256': m.digest(b'')} for name, pid in [('tp2-a', 4200001), ('tp2-b', 4200002)]}
    vr.register_deployment(data_root=data, fragment=source, model_fragment='model',
        receipt_path=receipt, instances=observations, session_id='resident', proc_root=w.proc_root)
    old_register = vr.register_deployment
    old_verify = vi.verify_vllm_process_identity
    monkeypatch.setattr(vr, 'register_deployment', lambda **kw: old_register(**kw, proc_root=w.proc_root))
    monkeypatch.setattr(vi, 'verify_vllm_process_identity', lambda *a, **kw: old_verify(*a, **kw, proc_root=w.proc_root))
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(m, 'quiescence', lambda instances: {'queues': {}, 'established_connections': []})
    target_pin = write_pin(repo/'configs/model/model.json', target)
    config = {'schema': m.SCHEMA, 'repo': str(repo), 'data': str(data), 'model': 'model',
        'session': 'resident', 'observations': write_pin(tmp_path/'observations.json', observations),
        'source_fragment': write_pin(tmp_path/'source.json', source),
        'target_fragment_sha256': target_pin['sha256'],
        'expected_source_sha256': m.inventory_sha256(snapshot(data)),
        'expected_old_budget': m.budget(source), 'expected_new_budget': m.budget(target),
        'out': str(tmp_path/'transition')}
    return w, config, source, target


def snapshot(data):
    return {str(p.relative_to(data/'serving')): p.read_bytes()
            for name in ('resident', 'tp2-a', 'tp2-b') for p in (data/'serving'/name).iterdir()
            if p.name in m.LEAVES}


def test_bidirectional_budget_change_preserves_real_live_receipts(world):
    w, config, source, target = world; data = Path(config['data'])
    before = snapshot(data)
    result = m.transition(config)
    assert result['status'] == 'COMPLETE' and result['no_engine_restart'] and result['new_model_calls'] == 0
    after = snapshot(data)
    for rel in before:
        if rel.endswith('session-receipt.json'): assert after[rel] == before[rel]
        else: assert json.loads(after[rel])['serving_geometry']['max_tokens'] == 16384
    assert not (data/'locks/serving-budget-transition.pending.json').exists()
    config['source_fragment'] = write_pin(w.root/'return-source.json', target)
    config['target_fragment_sha256'] = write_pin(Path(config['repo'])/'configs/model/model.json', source)['sha256']
    config['expected_source_sha256'] = m.inventory_sha256(after)
    config['expected_old_budget'], config['expected_new_budget'] = m.budget(target), m.budget(source)
    config['out'] = str(w.root/'return-transition')
    assert m.transition(config)['new_budget']['max_tokens'] == 4096
    assert snapshot(data) == before


@pytest.mark.parametrize('fault', ['lease', 'unknown_lease', 'source_hash', 'observation', 'old_budget',
                                  'target_hash', 'engine_change', 'stopped', 'process_dead', 'pending', 'unsafe_output'])
def test_preconditions_preserve_serving_and_all_unknown_evidence(world, fault):
    w, config, source, target = world; data = Path(config['data']); before = snapshot(data)
    lease = data/'locks/flight.lock'; lease.parent.mkdir(parents=True, exist_ok=True)
    if fault == 'lease': lease.write_text('{unknown owner}')
    elif fault == 'unknown_lease': lease.symlink_to(data/'missing-lease-owner')
    elif fault == 'source_hash': config['expected_source_sha256'] = '0'*64
    elif fault == 'observation':
        obs = json.loads(Path(config['observations']['path']).read_bytes()); obs['tp2-a']['pid'] = 999999
        config['observations'] = write_pin(w.root/'wrong-observations.json', obs)
    elif fault == 'old_budget': config['expected_old_budget']['max_tokens'] = 4097
    elif fault == 'target_hash': config['target_fragment_sha256'] = '0'*64
    elif fault == 'engine_change':
        target['server_args'] = [*target['server_args'], '--enforce-eager']
        config['target_fragment_sha256'] = write_pin(Path(config['repo'])/'configs/model/model.json', target)['sha256']
    elif fault == 'stopped': (data/'serving/tp2-a/session-stop.json').write_text('{}')
    elif fault == 'process_dead': (w.proc_root/'4200001/stat').unlink()
    elif fault == 'pending': (data/'locks/serving-budget-transition.pending.json').write_text('unknown previous transition')
    else: config['out'] = str(data/'serving/output')
    with pytest.raises((ValueError, RuntimeError)): m.transition(config)
    assert snapshot(data) == before
    assert not Path(config['out']).exists()
    if fault in ('lease', 'unknown_lease'): assert __import__('os').path.lexists(lease)
    if fault == 'pending': assert (data/'locks/serving-budget-transition.pending.json').read_text() == 'unknown previous transition'


def test_shared_guard_is_mutually_exclusive(world):
    _, config, _, _ = world; data = Path(config['data']); before = snapshot(data)
    with m.exclusive(data/'locks/generation.guard'):
        with pytest.raises(BlockingIOError): m.transition(config)
    assert snapshot(data) == before


def test_failed_registration_preserves_partial_outputs_and_restores_exact_original(world, monkeypatch):
    _, config, _, _ = world; data = Path(config['data']); before = snapshot(data)
    register = vr.register_deployment
    def fail(**kwargs):
        register(**kwargs)
        raise OSError('observed registration failure')
    monkeypatch.setattr(vr, 'register_deployment', fail)
    with pytest.raises(OSError, match='observed registration'): m.transition(config)
    out = Path(config['out']); failed = json.loads((out/'failed.json').read_bytes())
    assert failed['restored_original'] and failed['status'] == 'ROLLED_BACK'
    assert snapshot(data) == before
    assert len(list((out/'interrupted').glob('*/*.json'))) == 6
    assert not (data/'locks/serving-budget-transition.pending.json').exists()
    with pytest.raises(ValueError, match='fresh'): m.transition(config)


def test_unknown_postfailure_quiescence_keeps_pending_and_never_retries(world, monkeypatch):
    _, config, _, _ = world; data = Path(config['data']); before = snapshot(data)
    calls = []
    def quiet(instances):
        calls.append(1)
        if len(calls) > 1: raise OSError('metrics connection unknown')
        return {'queues': {}, 'established_connections': []}
    monkeypatch.setattr(m, 'quiescence', quiet)
    monkeypatch.setattr(vr, 'register_deployment', lambda **kw: (_ for _ in ()).throw(OSError('incomplete registration')))
    with pytest.raises(OSError, match='incomplete registration'): m.transition(config)
    out = Path(config['out']); failed = json.loads((out/'failed.json').read_bytes())
    assert failed['status'] == 'UNKNOWN' and not failed['restored_original']
    for rel, blob in before.items(): assert (out/'original'/rel).read_bytes() == blob
    pending = data/'locks/serving-budget-transition.pending.json'
    assert pending.exists()
    config['out'] = str(out.parent/'cannot-bypass-unknown')
    with pytest.raises(ValueError, match='unfinished'): m.transition(config)
    assert pending.exists() and not Path(config['out']).exists()


def test_new_unknown_lease_during_failure_is_never_removed(world, monkeypatch):
    _, config, _, _ = world; data = Path(config['data'])
    def fail(**kwargs):
        (data/'locks/flight.lock').write_text('foreign unknown lease')
        raise OSError('unexpected external lease')
    monkeypatch.setattr(vr, 'register_deployment', fail)
    with pytest.raises(OSError): m.transition(config)
    assert (data/'locks/flight.lock').read_text() == 'foreign unknown lease'
    assert (data/'locks/serving-budget-transition.pending.json').exists()
    assert json.loads((Path(config['out'])/'failed.json').read_bytes())['status'] == 'UNKNOWN'


def test_interrupted_registration_rolls_back_and_keeps_original_exception(world, monkeypatch):
    _, config, _, _ = world; data = Path(config['data']); before = snapshot(data)
    monkeypatch.setattr(vr, 'register_deployment', lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): m.transition(config)
    assert snapshot(data) == before
    failed = json.loads((Path(config['out'])/'failed.json').read_bytes())
    assert failed['error_type'] == 'KeyboardInterrupt' and failed['restored_original']


def test_failed_rollback_never_claims_restoration_or_drops_pending(world, monkeypatch):
    _, config, _, _ = world; data = Path(config['data']); before = snapshot(data)
    monkeypatch.setattr(vr, 'register_deployment', lambda **kw: (_ for _ in ()).throw(OSError('register failed')))
    monkeypatch.setattr(m.os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('rollback write failed')))
    with pytest.raises(OSError, match='register failed'): m.transition(config)
    failed = json.loads((Path(config['out'])/'failed.json').read_bytes())
    assert failed['status'] == 'UNKNOWN' and not failed['restored_original']
    assert failed['rollback_error'] == 'rollback write failed'
    assert (data/'locks/serving-budget-transition.pending.json').exists()
    for rel, blob in before.items(): assert (Path(config['out'])/'original'/rel).read_bytes() == blob


def test_changed_observation_during_quiescence_refused_before_any_identity_write(world, monkeypatch):
    _, config, _, _ = world; data = Path(config['data']); before = snapshot(data)
    def quiet(instances):
        Path(config['observations']['path']).write_text('{}')
        return {'queues': {}, 'established_connections': []}
    monkeypatch.setattr(m, 'quiescence', quiet)
    with pytest.raises(ValueError, match='input changed'): m.transition(config)
    assert snapshot(data) == before and not Path(config['out']).exists()


def metrics(tmp_path, content):
    (tmp_path/'net').mkdir(); (tmp_path/'net/tcp').write_text('header\n')
    (tmp_path/'net/tcp6').write_text('header\n')
    return lambda *a, **k: io.BytesIO(content.encode())


@pytest.mark.parametrize('content', ['', 'vllm:num_requests_running 0\n',
    'vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n',
    'vllm:num_requests_running NaN\nvllm:num_requests_waiting 0\n'])
def test_missing_active_or_nan_metrics_refuse_unknown_state(tmp_path, content):
    opener = metrics(tmp_path, content)
    with pytest.raises(ValueError): m.quiescence([{'host': '127.0.0.1', 'port': 9123}], proc_root=tmp_path, opener=opener)


def test_generic_quiescence_checks_declared_endpoints_and_established_connections(tmp_path):
    opener = metrics(tmp_path, 'vllm:num_requests_running{model="m"} 0\nvllm:num_requests_waiting 0\n')
    instances = [{'host': '127.0.0.1', 'port': 9123}]
    assert set(m.quiescence(instances, proc_root=tmp_path, opener=opener)['queues']) == {'9123'}
    (tmp_path/'net/tcp').write_text('header\n0: 0100007F:23A3 0100007F:EEEE 01 extra\n')
    with pytest.raises(ValueError, match='connection'): m.quiescence(instances, proc_root=tmp_path, opener=opener)
