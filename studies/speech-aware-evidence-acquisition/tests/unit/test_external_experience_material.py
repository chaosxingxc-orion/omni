"""Invalid material jobs must not publish templates or partial workload files."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
import external_experience_material as material
from test_tfgrpo_external_release import material as release_material


def job(kwargs):
    repo, data = kwargs['repo'], kwargs['data']
    documents = {
        'configs/model/frozen-actor.json': {'runtime_receipt': 'receipts/runtime.json'},
        'configs/deployment/profile.json': {'model_fragment': 'frozen-actor', 'admitted_level': 4,
            'instances': [{'name': 'a', 'cards': [4, 5]}]},
        'docs/receipts/runtime.json': {'deployment': {'instances': [{'instance_id': 'a'}]}}}
    for rel, doc in documents.items():
        path = repo / rel; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
    return {'schema': 'saea-external-experience-material-job-v1', 'source_kind': 'external-tfgrpo-v1',
        'name': 'unit-pair', 'run_id': 'UNIT-PAIR',
        'release': {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
        'runtime_binding': {'model_fragment': 'frozen-actor', 'service_session': 'session',
                            'profile': 'profile', 'runtime_receipt': 'receipts/runtime.json'},
        'execution': {'admitted_level': 2}, 'params_path': str(data/'params.json'),
        'measurement_path': str(data/'measurement.json'),
        'material_relpath': 'configs/evaluation/unit-material.json', 'workload_path': str(data/'workload.json')}


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_material_build_pins_original_release_and_real_workload(release_material, monkeypatch):
    kwargs, candidate, *_ = release_material
    repo, data = kwargs['repo'], kwargs['data']; cfg = job(kwargs)
    monkeypatch.setattr(material, 'measure', lambda *args: {'token_counts': {'template': 10}})
    result = material.build(cfg)
    assert result['new_model_calls'] == 0 and result['rows'] == 1
    workload = json.loads((data/'workload.json').read_bytes())
    assert material.ExternalExperienceWorkload(workload).runtime_binding(repo) == cfg['runtime_binding']
    params = json.loads((data/'params.json').read_bytes())
    assert params['candidate']['sha256'] == hashlib.sha256(kwargs['candidate_path'].read_bytes()).hexdigest()
    assert list(candidate['items'].values())[0] in (repo/'configs/probes/external.txt').read_text()


@pytest.mark.parametrize('fault', ['source_kind', 'reused_output', 'escaping_material', 'escaping_workload',
    'width', 'boolean_width', 'lane_width', 'extra_execution', 'runtime_missing', 'runtime_name',
    'runtime_receipt_escape', 'runtime_model', 'membership', 'duplicate_cards', 'release_model',
    'name', 'run_id', 'unsorted_consumers'])
def test_invalid_configuration_has_no_prepare_or_publish_side_effects(release_material, monkeypatch, fault):
    kwargs, *_ = release_material; repo, data = kwargs['repo'], kwargs['data']; cfg = job(kwargs)
    if fault == 'source_kind': cfg['source_kind'] = 'unknown'
    elif fault == 'reused_output': (data/'params.json').write_text('old')
    elif fault == 'escaping_material': cfg['material_relpath'] = '../escape.json'
    elif fault == 'escaping_workload': cfg['workload_path'] = str(data.parent/'escape.json')
    elif fault == 'width': cfg['execution']['admitted_level'] = 97
    elif fault == 'boolean_width': cfg['execution']['admitted_level'] = True
    elif fault == 'lane_width': cfg['execution']['admitted_level'] = 8
    elif fault == 'extra_execution': cfg['execution']['unreviewed'] = True
    elif fault == 'runtime_missing': del cfg['runtime_binding']['service_session']
    elif fault == 'runtime_name': cfg['runtime_binding']['profile'] = '../other'
    elif fault == 'runtime_receipt_escape': cfg['runtime_binding']['runtime_receipt'] = '../../outside.json'
    elif fault == 'runtime_model':
        (repo/'configs/model/frozen-actor.json').write_text(json.dumps({'runtime_receipt': 'wrong.json'}))
    elif fault in ('membership', 'duplicate_cards'):
        p = repo/'configs/deployment/profile.json'; doc = json.loads(p.read_text())
        if fault == 'membership': doc['instances'][0]['name'] = 'other'
        else: doc['instances'][0]['cards'] = [4, 4]
        p.write_text(json.dumps(doc))
    elif fault == 'release_model': cfg['release']['model_id'] = 'another-model'
    elif fault == 'name': cfg['name'] = 'UpperCase'
    elif fault == 'run_id': cfg['run_id'] = 'lowercase'
    else: cfg['release']['consumer_ids'] = ['slurp:train:z', 'slurp:train:a']
    before = snapshot(repo.parent); called = []
    def forbidden(*args, **kwargs):
        called.append(True); raise AssertionError('adapter or publication reached before validation')
    # Importing an adapter is the first step toward prepare; this also rejects
    # a future refactor that calls prepare but happens not to write a file.
    monkeypatch.setattr(material.importlib, 'import_module', forbidden)
    monkeypatch.setattr(material, 'publish_new', forbidden)
    monkeypatch.setattr(material, 'measure', forbidden)
    with pytest.raises(ValueError): material.build(cfg)
    assert called == []
    assert snapshot(repo.parent) == before


def test_material_contract_design_is_checked_before_publication(release_material, monkeypatch):
    kwargs, *_ = release_material; cfg = job(kwargs)
    cfg['evaluation_design']='contract-control-v1'
    monkeypatch.setattr(material.importlib,'import_module',lambda *a:pytest.fail('adapter reached'))
    before=snapshot(kwargs['repo'].parent)
    with pytest.raises(ValueError,match='design differ'): material.build(cfg)
    assert snapshot(kwargs['repo'].parent)==before
