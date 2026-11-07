import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'analysis'))
import p_only_workload as p


def pin(path, root):
    return {'relpath': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_axes_preserve_parent_and_reject_changed_template(tmp_path, monkeypatch):
    parent = {'native_episode': {'perception_only': True},
              'thinking_policy': {'perception': {'enable_thinking': False}},
              'native_contact_fit': {'run_id': 'old', 'attempt_id': 'old', 'contact_cap': 6480},
              'phase_material': {'perception': {'instruction_template': 'probes/old.txt'}},
              'knowledge_stores': {'intent-boundaries': {'phase_supply': {'perception': {'k': 8, 'fuse_coverage': {'m': 8, 'top_l': 8}}}}},
              'prepared_phase_supply': {'relpath': 'old', 'sha256': 'old'}}
    src = tmp_path / 'parent.json'; src.write_text(json.dumps(parent))
    job = object.__new__(p.PerceptionOnlyWorkload)
    job.NAME = 'p-test'; job.RUN = 'P-TEST'; job.ROWS = 1080; job.DATASET = 'fold'; job.ROSTER = 'roster'
    from tfrl_hardset_a0_registration import scientific_hash
    science = copy.deepcopy(parent); science['native_episode'].pop('perception_only'); science['native_contact_fit'].update(run_id='old', attempt_id='old')
    registration = tmp_path / 'registration.json'
    original = {'sample_ids':['id'], 'population':{'roster':{'relpath':'r','sha256':'x'}}, 'material':{}, 'runtime_binding':{}, 'execution':{}}
    registration.write_text(json.dumps({'workload_kind':'intent-slot-node-ablation', 'job_config':original, 'fit_run_id':'old', 'fit_attempt_id':'old', 'scientific_baseline_sha256':scientific_hash(science)}))
    job.config = {'parent_registration':pin(registration,tmp_path), 'material':{}, 'runtime_binding':{}, 'execution':{}, 'parent_baseline': pin(src, tmp_path), 'axis': {'kind': 'control'},
                  'population': {'roster': {'relpath': 'r', 'sha256': 'x'}}, 'sample_ids': ['id']}
    roster = {'items': [{'audio_seconds': 1}], 'split_identity_hash': 'split', 'roster_id': 'r'}
    monkeypatch.setattr(job, 'material_checked', lambda *args: ({}, {'roster': copy.deepcopy(roster), 'view': {'split_name': 'fold'}}))
    monkeypatch.setattr(job, 'runtime_binding', lambda *args: {'model_fragment': 'm', 'service_session': 's'})
    monkeypatch.setattr(p, 'read_asset', lambda *args: {'cells': [{}], 'population': {}, 'execution_plan': {}, 'stage3_measurement': {}})
    control = job.derive(tmp_path, tmp_path, {})[0]
    parent_bytes = registration.read_bytes()
    job.config['sample_ids'] = ['changed']
    with pytest.raises(ValueError, match='source differs'):
        job.derive(tmp_path, tmp_path, {})
    job.config['sample_ids'] = ['id']
    assert registration.read_bytes() == parent_bytes
    expected = copy.deepcopy(parent); expected.pop('prepared_phase_supply'); expected['native_contact_fit'].update(run_id='P-TEST-FIT', attempt_id='fit-v1')
    assert control == expected
    job.config['axis'] = {'kind': 'breadth', 'value': 12}
    breadth = job.derive(tmp_path, tmp_path, {})[0]
    expected['knowledge_stores']['intent-boundaries']['phase_supply']['perception'].update(k=12, fuse_coverage={'m': 12, 'top_l': 12})
    assert breadth == expected
    template = tmp_path / 'configs/probes/new.txt'; template.parent.mkdir(parents=True); template.write_text('new')
    patch = tmp_path / 'patch.json'; patch.write_text(json.dumps([{'op':'test','path':'/phase_material/perception/instruction_template','value':'probes/old.txt'}, {'op':'replace','path':'/phase_material/perception/instruction_template','value':'probes/new.txt'}]))
    job.config['axis'] = {'kind':'instruction', 'patch':pin(patch,tmp_path), 'template':pin(template,tmp_path)}
    changed = job.derive(tmp_path,tmp_path,{})[0]
    assert changed['phase_material']['perception']['instruction_template'] == 'probes/new.txt'
    template.write_text('tampered')
    with pytest.raises(ValueError, match='template pin'):
        job.derive(tmp_path,tmp_path,{})


def test_explicit_binding_adaptation_pins_science_and_preserves_history(tmp_path):
    def write(rel, value):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return pin(path, tmp_path)

    bindings = [dict(model_fragment=name, profile=name, service_session=name,
                     runtime_receipt='receipts/'+name+'.json') for name in ('old', 'new')]
    models = [dict(model_lock_key='frozen', base_url=name, runtime_receipt=b['runtime_receipt'],
                   decoding_params={'temperature': 0, 'max_tokens': 4096},
                   server_args=['--max-model-len', '65536']) for name, b in zip(('old', 'new'), bindings)]
    receipt = {'model_lock_key':'frozen', 'model':{'files':[
        {'path':'models/frozen/model.safetensors', 'bytes':42, 'sha256':'a'*64},
        {'path':'models/frozen/config.json', 'bytes':5, 'sha256':'b'*64}]}}
    policy = {'schema':'saea-p-only-runtime-binding-adaptation-v1', 'comparison_scope':'same-model-and-request'}
    for label, binding, model in zip(('parent','current'), bindings, models):
        policy[label+'_model'] = write('configs/model/'+binding['model_fragment']+'.json', model)
        policy[label+'_receipt'] = write('docs/'+binding['runtime_receipt'], receipt)
    job = object.__new__(p.PerceptionOnlyWorkload)
    job.config = {'runtime_binding':copy.deepcopy(bindings[0])}
    job.check_parent_binding(tmp_path, bindings[0])
    job.config['runtime_binding'] = bindings[1]
    with pytest.raises(ValueError, match='source differs'):
        job.check_parent_binding(tmp_path, bindings[0])
    job.config['runtime_binding_adaptation'] = policy
    before = copy.deepcopy(bindings[0])
    job.check_parent_binding(tmp_path, bindings[0])
    assert bindings[0] == before
    for key, value in [('decoding_params', {'temperature':0.1, 'max_tokens':4096}),
                       ('server_args', ['--max-model-len', '32768'])]:
        changed = copy.deepcopy(models[1]); changed[key] = value
        policy['current_model'] = write('configs/model/new.json', changed)
        with pytest.raises(ValueError, match='science or weights'):
            job.check_parent_binding(tmp_path, bindings[0])
    policy['current_model'] = write('configs/model/new.json', models[1])
    changed = copy.deepcopy(receipt); changed['model']['files'][0]['sha256'] = 'c'*64
    policy['current_receipt'] = write('docs/receipts/new.json', changed)
    with pytest.raises(ValueError, match='science or weights'):
        job.check_parent_binding(tmp_path, bindings[0])
