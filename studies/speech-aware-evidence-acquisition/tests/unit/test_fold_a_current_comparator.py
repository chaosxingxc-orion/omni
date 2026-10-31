"""The bootstrap comparator must preserve current A0 recall and serving."""
import copy
import json
from pathlib import Path
import shutil
import sys
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/flights/nx'))
import fold_a_current_comparator as derive
from speech_aware_evidence_acquisition.core.config import compose
from speech_aware_evidence_acquisition.rails import consumption_module as cm


@pytest.fixture
def source():
    spec = derive.read(REPO / derive.SPEC)
    base = derive.read(REPO / 'configs/baseline' / (spec['baseline'] + '.json'))
    exp = derive.read(REPO / 'configs/experiment' / (spec['cell'] + '.json'))
    p = derive.inputs
    pins = {name: str(i) * 64 for i, name in enumerate((p.RECEIPT,
        derive.read(REPO / p.PROTOCOL)['query_rows_relpath'], p.ALIAS_RECEIPT,
        p.SEGMENTS, p.CONTENT, p.PROVENANCE), 1)}
    return base, exp, pins


def test_current_module_compiles_and_retains_recall_model_and_session(source, tmp_path):
    base, exp, pins = source
    before = copy.deepcopy((base, exp))
    b, e = derive.derive(REPO, base, exp, pins)
    assert (base, exp) == before
    assert b['policy'] == base['policy'] == 'phase-native-module-v1-c17'
    assert b['native_episode']['roles'] == base['native_episode']['roles']
    assert b['knowledge_stores']['slot-contracts'] == base['knowledge_stores']['slot-contracts']
    assert b['knowledge_stores']['train-exemplars']['snapshot'] == base['knowledge_stores']['train-exemplars']['snapshot']
    for store in ('train-exemplars', 'intent-boundaries'):
        phase = 'solver' if store == 'train-exemplars' else 'perception'
        assert b['knowledge_stores'][store]['phase_supply'][phase]['facet_source'] == base['knowledge_stores'][store]['phase_supply'][phase]['facet_source']
    solver = b['knowledge_stores']['train-exemplars']['phase_supply']['solver']
    original = base['knowledge_stores']['train-exemplars']['phase_supply']['solver']
    assert {k:v for k,v in solver.items() if k!='query_source'} == {k:v for k,v in original.items() if k!='query_source'}
    assert solver['query_source']['sha256'] == pins[derive.inputs.CONTENT]
    assert solver['query_source']['relpath'] == derive.inputs.CONTENT
    assert 'agreement_source' not in json.dumps(b)
    assert e['cells'][0]['model_fragment'] == exp['cells'][0]['model_fragment']
    assert e['service_session'] == exp['service_session']
    assert e['cells'][0]['allowed_execution_stages'] == ['check']
    assert e['cells'][0]['readiness']['state'] == 'held'
    assert len(e['populations'][0]['ids']) == 1080
    assert e['execution_plan']['planned_speech_audio_seconds'] == 104874
    assert all('FOLD-B' not in p['relpath'] and 'fold-b' not in p['relpath'] for p in b['preparation_source_receipts'])
    cm.verify_declared_module(b['native_episode']['consumption_module'], b, repo_root=REPO)
    # Compose real model/dataset fragments with private new configuration files.
    shutil.copytree(REPO / 'configs', tmp_path / 'configs')
    (tmp_path / f'configs/baseline/{derive.CELL}.json').write_bytes(derive.raw(b))
    (tmp_path / f'configs/experiment/{derive.CELL}.json').write_bytes(derive.raw(e))
    cell = e['cells'][0]
    cfg = compose(tmp_path, cell['model_fragment'], cell['dataset_fragment'], derive.CELL, derive.CELL)
    cm.verify_declared_module(cfg.values['native_episode']['consumption_module'], cfg.values, repo_root=REPO)


@pytest.mark.parametrize('fault', ['missing', 'zero', 'nonhex', 'recall-drift', 'module-drift'])
def test_unpinned_or_inconsistent_source_refused(source, fault):
    base, exp, pins = source
    if fault == 'missing': pins.pop(derive.inputs.CONTENT)
    elif fault == 'zero': pins[derive.inputs.CONTENT] = '0' * 64
    elif fault == 'nonhex': pins[derive.inputs.CONTENT] = 'z' * 64
    elif fault == 'recall-drift': base['knowledge_stores']['train-exemplars']['phase_supply']['solver']['k'] = 3
    else: base['native_episode']['consumption_module']['lock'] = 'v2.2'
    with pytest.raises((ValueError, cm.ConsumptionModuleConfigError)): derive.derive(REPO, base, exp, pins)


@pytest.mark.parametrize('fault', [None, 'missing-parent', 'source-head', 'model-drift', 'vector-drift',
    'queued', 'queue-failed', 'queue-exit', 'queue-id', 'parent-commit', 'parent-run',
    'retrieval-commit', 'spec-hash', 'path-escape'])
def test_writer_binds_actual_completed_source_and_fresh_outputs(source, tmp_path, monkeypatch, fault):
    base, exp, _ = source
    repo, src, data = tmp_path / 'repo', tmp_path / 'source', tmp_path / 'data'
    shutil.copytree(REPO / 'configs', repo / 'configs')
    for name in (derive.inputs.ROSTER, derive.inputs.DATASET, derive.SPEC,
                 'docs/results/2026-09-30-slurp-a2-comp1b-dev2000-read.md',
                 'scripts/flights/nx/l338_port_pins.json'):
        dest = repo / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO / name, dest)
    spec = derive.read(REPO / derive.SPEC)
    model_rel = 'configs/model/' + exp['cells'][0]['model_fragment'] + '.json'
    def put(path, doc):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(derive.raw(doc))
    put(src / derive.SPEC, spec)
    put(src / f"configs/baseline/{spec['baseline']}.json", base)
    put(src / f"configs/experiment/{spec['cell']}.json", exp)
    model = src / model_rel; model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / model_rel, model)
    protocol = derive.read(repo / derive.inputs.PROTOCOL)
    p = derive.inputs
    pins = {}
    for name in (p.RECEIPT, protocol['query_rows_relpath'], p.ALIAS_RECEIPT, p.SEGMENTS, p.CONTENT, p.PROVENANCE):
        put(data / name, {'synthetic_sealed_fixture': name})
        pins[name] = derive.sha(data / name)
    put(data / derive.SOURCE_COMPLETE, {'commit': 'actual-head', 'rows': 1080,
        'score_surface': 'sealed-final-delivery', 'run_id': spec['run_id']})
    put(data / 'work/codex/agent-after-recovered-direct-v1/status.json',
        {'state': 'running' if fault=='missing-parent' else 'done'})
    put(data / derive.RETRIEVAL_COMPLETE, {'rows': 1080, 'values_unchanged': True, 'pins': pins})
    put(data / 'work/codex/fold-a-retrieval-after-encoding-v1/status.json', {'state': 'done'})
    monkeypatch.setattr(derive.subprocess, 'check_output', lambda args, **kw:
        ('wrong' if fault=='source-head' else 'actual-head') if 'rev-parse' in args else b'')
    binding = None
    queued = fault in {'queued', 'queue-failed', 'queue-exit', 'queue-id', 'parent-commit',
                      'parent-run', 'retrieval-commit', 'spec-hash', 'path-escape'}
    if queued:
        # New queue has no historical follower/status directories. Both complete
        # records and the source spec live at explicitly bound successor paths.
        spec_new = 'docs/readiness/successor-v5/spec.json'
        put(src / spec_new, spec)
        new_source = 'work/codex/successor-v5/campaign/complete.json'
        new_retrieval = 'work/codex/retrieval-v5/complete.json'
        put(data / new_source, derive.read(data / derive.SOURCE_COMPLETE))
        put(data / new_retrieval, {**derive.read(data / derive.RETRIEVAL_COMPLETE),
            'schema': 'saea-fold-a-retrieval-inputs-v1', 'source_commit': 'retrieval-head'})
        shutil.rmtree(data / 'work/codex/agent-after-recovered-direct-v1')
        shutil.rmtree(data / 'work/codex/fold-a-retrieval-after-encoding-v1')
        jobs = {key: {'status': 'done', 'exit': 0} for key in
                ('a0-campaign', 'a0-paired', 'fold-a-encoding', 'fold-a-retrieval')}
        if fault == 'queue-failed': jobs['a0-paired']['status'] = 'failed'
        if fault == 'queue-exit': jobs['fold-a-retrieval']['exit'] = 1
        put(data / 'work/queues/v5/state.json', {'schema': 'saea-card-queue-state-v1',
            'queue_id': 'wrong' if fault == 'queue-id' else 'v5', 'jobs': jobs})
        binding = {'schema': 'saea-fold-a-comparator-parent-v1', 'source_spec': spec_new,
            'source_spec_sha256': derive.sha(src / spec_new),
            'source_commit': 'actual-head', 'source_run': spec['run_id'],
            'retrieval_commit': 'retrieval-head', 'source_complete': new_source,
            'retrieval_complete': new_retrieval, 'queue_state': 'work/queues/v5/state.json',
            'queue_id': 'v5'}
        changes = {'parent-commit': ('source_commit', 'wrong'),
            'parent-run': ('source_run', 'wrong'), 'retrieval-commit': ('retrieval_commit', 'wrong'),
            'spec-hash': ('source_spec_sha256', '0'*64),
            'path-escape': ('source_complete', '../outside/complete.json')}
        if fault in changes:
            key, value = changes[fault]; binding[key] = value
    if fault=='model-drift': (repo / model_rel).write_bytes((repo / model_rel).read_bytes()+b' ')
    if fault=='vector-drift': (data / p.CONTENT).write_bytes(b'changed')
    if fault not in (None, 'queued'):
        with pytest.raises(ValueError): derive.write(repo, src, data, data/'out', binding)
        assert not (repo / f'configs/baseline/{derive.CELL}.json').exists()
    else:
        record = derive.write(repo, src, data, data/'out', binding)
        if queued:
            assert record['parent_binding'] == binding
            assert record['queue_state_sha256'] == derive.sha(data / binding['queue_state'])
        assert record['source_commit']=='actual-head'
        assert record['source_files'][model_rel]==derive.sha(model)
        assert record['allowed_execution_stages']==['check']
        assert (repo / f'configs/baseline/{derive.CELL}.json').is_file()
        with pytest.raises(ValueError, match='fresh derivation'):
            derive.write(repo, src, data, data/'other-out', binding)
