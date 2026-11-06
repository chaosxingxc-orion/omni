import copy
import hashlib
import json
from pathlib import Path
import pytest
import psd_tfgrpo_adapter as adapter
from test_intent_slot_workload import config, ROOT
import intent_slot_workload as workload


def pin(root, name, value):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'relpath':name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize('target,role', [('slot','slot'),('decision','solver')])
def test_node_input_and_frozen_scope(tmp_path, monkeypatch, target, role):
    from speech_aware_evidence_acquisition.rails import tfgrpo_external_release as release
    from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
    from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs, render_static_instruction
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    repo=tmp_path/'repo'; data=tmp_path/'data'; repo.mkdir(); data.mkdir()
    original_read=workload.read_asset
    monkeypatch.setattr(workload,'read_asset',lambda repo,rel:original_read(ROOT,rel))
    monkeypatch.setattr(workload.NodeWorkload,'material_checked',lambda *a: ({'source_roster':{}},{'roster':{'roster_id':'fixture','items':[{'audio_seconds':1}],'split_identity_hash':'a'*64}, 'view':{'items':[{'sample_id':'slurp:train:2:a','audio_sha256':'b'*64}]}}))
    monkeypatch.setattr(workload.NodeWorkload,'runtime_binding',lambda *a:{'model_fragment':'bf16-thinking-tfgrpo-tp4-ctx48k-v1','service_session':'fixture'})
    # Isolate the source replay dependency here; its actual verifier has its own tests.
    monkeypatch.setattr(consumption,'validate_params',lambda p:p)
    monkeypatch.setattr(release,'verify_source',lambda *a,**k:None)
    monkeypatch.setattr(release,'render',lambda c,b,ids:b+' old')
    c=config(); c['node_arm']='enhanced-p-slot-decision'; c['pipeline']={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1'}
    base=workload.NodeWorkload(c).derive(repo,data,{})[0]
    for r in ('slot','solver'):
        rel=base['phase_material'][r]['instruction_template']; path=repo/'configs'/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((ROOT/'configs'/rel).read_bytes())
    old=repo/'configs/probes/old.txt'; old.write_text('base old')
    oldbase=repo/'configs/probes/oldbase.txt'; oldbase.write_text('base')
    def fp(path):return {'relpath':path.relative_to(repo/'configs').as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    cp=pin(data,'candidate.json',{'sha256':'c'*64,'items':{'x':'old experience'},'learned_query_ids':['slurp:train:1:a']})
    ds=pin(repo,'dataset.json',{'items':[{'sample_id':'slurp:train:1:a','audio_sha256':'a'*64}]})
    rp=pin(data,'release.json',{'learning_dataset':ds})
    source=pin(data,'source.json',{'source_kind':adapter.KINDS[0], 'candidate':cp,'candidate_release':rp,'model_id':'bf16-thinking-tfgrpo-tp4-ctx48k-v1','base_template':fp(oldbase),'template':fp(old),'consumer_ids':['old']})
    mapping=pin(data,'mapping.json',{'schema':'saea-psd-tfgrpo-semantic-map-v1','target':target,'candidate_sha256':'c'*64,'items':{'x':'inspect available evidence'},'rationale':'explicit adaptation'})
    d={'schema':adapter.SCHEMA,'target':target,'source':source,'mapping':mapping,'template':{'relpath':'configs/probes/new.txt','sha256':''}}
    view={'split_name':'fixture','items':[{'sample_id':'slurp:train:2:a','audio_sha256':'b'*64}]}
    original=(repo/'configs'/base['phase_material'][role]['instruction_template']).read_text()
    text=adapter.verify_and_render(repo,data,d,['slurp:train:2:a'],view,'bf16-thinking-tfgrpo-tp4-ctx48k-v1',original)
    path=repo/d['template']['relpath']; path.write_text(text); d['template']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    result=adapter.apply(base,repo,data,d,['slurp:train:2:a'],view,'bf16-thinking-tfgrpo-tp4-ctx48k-v1')
    expected=copy.deepcopy(base); expected['phase_material'][role]['instruction_template']='probes/new.txt'
    assert result==expected
    # Exercise the real NodeWorkload entry while isolating fixed-roster I/O.
    c['population']={'dataset':'slurp-speech-native-evo-fold-a-1080','dataset_sha256':'x','roster':{},'source_baseline':{}}
    c['sample_ids']=['slurp:train:2:a']
    monkeypatch.setattr(workload.NodeWorkload, 'material_checked', lambda *a: ({'source_roster':{}}, {'roster':{'roster_id':'fixture','items':[{'audio_seconds':1}], 'split_identity_hash':'a'*64}, 'view':view, 'source_baseline':original_read(ROOT, workload.observation.SOURCE_BASE)}))
    # Fold derivation has its own frozen source conversion; compare with and without the adapter.
    unmodified=workload.NodeWorkload(c).derive(repo,data,{})[0]
    node_original=(repo/'configs'/unmodified['phase_material'][role]['instruction_template']).read_text()
    path.write_text(adapter.verify_and_render(repo,data,d,c['sample_ids'],view,'bf16-thinking-tfgrpo-tp4-ctx48k-v1',node_original))
    d['template']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    c['tfgrpo_experience']=d
    actual=workload.NodeWorkload(c).derive(repo,data,{})[0]
    node_expected=copy.deepcopy(unmodified); node_expected['phase_material'][role]['instruction_template']='probes/new.txt'
    assert actual==node_expected
    specs=load_phase_material_specs(result,default_phase_program(result['policy']))
    assert 'inspect available evidence' in render_static_instruction(specs[role],repo,{},data)
    if target=='decision': assert 'Entities are copied from Slot' in text
    (data/'mapping.json').write_text('{}')
    with pytest.raises(ValueError): adapter.apply(base,repo,data,d,['slurp:train:2:a'],view,'bf16-thinking-tfgrpo-tp4-ctx48k-v1')
    with pytest.raises(ValueError): adapter.verify_and_render(repo,data,{**d,'target':'legacy'},[],view,'bf16-thinking-tfgrpo-tp4-ctx48k-v1',original)


def test_old_workload_and_v2_gate():
    c=config(); assert 'tfgrpo_experience' not in workload.NodeWorkload(c).config
    c['tfgrpo_experience']={}
    with pytest.raises(ValueError,match='fixed fold'): workload.NodeWorkload(c)
