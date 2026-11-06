import copy
import hashlib
import json

import pytest
import psd_knowledge_adapter as adapter
import intent_slot_workload as workload
from test_external_slot_candidate import fixture
from test_intent_slot_workload import ROOT, config
from speech_aware_evidence_acquisition.rails.errors import RailsError
from speech_aware_evidence_acquisition.rails.knowledge_value_views import KnowledgeValueRenderer
from speech_aware_evidence_acquisition.rails.knowledge_stores import load_knowledge_store_binding


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def assets(tmp_path, monkeypatch):
    original, base, contract, dynamic, candidate, custody, forbidden = fixture()
    inputs = dict(zip(('original','base','contract','dynamic','forbidden_sample_ids'),
                      (original,base,contract,dynamic,forbidden)))
    pins = {k:write(tmp_path/(k+'.json'),v) for k,v in inputs.items()}
    cfg = {'schema':'saea-external-slot-candidate-config-v1', **pins,
           'teacher':{}, 'candidate_index':0, 'output':'unused'}
    compiler = write(tmp_path/'compiler.json', cfg)
    monkeypatch.setattr(adapter, 'verify_response', lambda pin, value: ({'candidates':[candidate]}, custody))
    data = tmp_path/'data'; data.mkdir()
    descriptor, audit = adapter.publish(data, compiler, 'knowledge/child')
    return data, descriptor, audit, original, inputs, cfg


def test_publication_source_and_versions_are_isolated(assets):
    data, desc, audit, original, _, _ = assets
    assert audit['changed_value_ids'] == ['slot-contract:date']
    assert audit['snapshot_sha256'] != original['snapshot_sha256']
    second, _ = adapter.publish(data, desc['compiler'], 'knowledge/sibling')
    assert desc['release']['relpath'] != second['release']['relpath']
    assert desc['snapshot']['sha256'] == second['snapshot']['sha256']
    with pytest.raises(FileExistsError):
        adapter.publish(data, desc['compiler'], 'knowledge/child')


def test_node_derivation_and_actual_supply_render(assets, tmp_path, monkeypatch):
    data, desc, _, original, _, _ = assets
    source_read = workload.read_asset
    monkeypatch.setattr(workload, 'read_asset', lambda repo, rel: source_read(ROOT, rel))
    monkeypatch.setattr(workload.NodeWorkload, 'material_checked', lambda *a: ({'source_roster':{}}, {
        'source_baseline': source_read(ROOT, 'configs/evaluation/fold-baseline-v1/fold-a-source.json'),
        'roster':{'roster_id':'fixture','items':[{'audio_seconds':1}], 'split_identity_hash':'a'*64}, 'view':{'split_name':'fixture'}}))
    monkeypatch.setattr(workload.NodeWorkload, 'runtime_binding', lambda *a: {'model_fragment':'bf16-thinking-tfgrpo-tp4-ctx48k-v1','service_session':'fixture'})
    c=config(); c['node_arm']='enhanced-p-slot-decision'
    c['pipeline']={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1'}
    c['population']={'dataset':'slurp-speech-native-evo-fold-a-1080','dataset_sha256':'a'*64,'roster':{},'source_baseline':{}}
    original_lock = workload.NodeWorkload.derive
    # Patch only the source's slot parent identity, preserving the real derive path.
    orig_checked = workload.NodeWorkload.material_checked
    def source(*args):
        material, checked = orig_checked(*args)
        checked['source_baseline']['knowledge_stores']['slot-contracts']['snapshot']['sha256']=original['snapshot_sha256']
        checked['source_baseline']['prepared_phase_supply']={'stale':True}
        return material, checked
    monkeypatch.setattr(workload.NodeWorkload, 'material_checked', source)
    baseline=original_lock(workload.NodeWorkload(c), ROOT, data, {})[0]
    c['slot_knowledge']=desc
    derived=original_lock(workload.NodeWorkload(c), ROOT, data, {})[0]
    expected=copy.deepcopy(baseline)
    expected['knowledge_stores']['slot-contracts']['release']=desc['release']
    expected['knowledge_stores']['slot-contracts']['snapshot']=desc['snapshot']
    assert derived == expected
    assert 'prepared_phase_supply' not in derived
    assert derived['native_episode']['intent_slot_pipeline']['node_rails']=='sdk-handoff-v1'
    # Use the real named-store loader and renderer on the derived store, preserving
    # its actual slot and solver supply definitions and candidate routing.
    single={'knowledge_stores':{'slot-contracts':derived['knowledge_stores']['slot-contracts']}}
    binding=load_knowledge_store_binding(single, data_root=data)
    value_id='slot-contract:date'
    blocks=KnowledgeValueRenderer(binding.composite_snapshot).assemble(('slot-contracts::'+value_id,))
    assert blocks[0]['text'] == next(v['content'] for v in json.loads((data/desc['snapshot']['relpath']).read_text())['values'] if v['value_id']==value_id)
    assert blocks[0]['text'] != next(v['content'] for v in original['values'] if v['value_id']==value_id)
    assert 'slot' in derived['knowledge_stores']['slot-contracts']['candidate_supply']


@pytest.mark.parametrize('target',['compiler','base','release','snapshot','source'])
def test_rejects_tampered_sources_and_publications(assets,target):
    data, desc, _, original, _, cfg = assets
    base={'knowledge_stores':{'slot-contracts':{'store_role':'slot-contracts', 'snapshot':{'sha256':original['snapshot_sha256']}}}}
    if target=='compiler': path=desc['compiler']['path']
    elif target=='base': path=cfg['base']['path']
    elif target=='release': path=data/desc['release']['relpath']
    elif target=='snapshot': path=data/desc['snapshot']['relpath']
    else:
        release=json.loads((data/desc['release']['relpath']).read_text())
        path=data/release['source']['relpath']
    from pathlib import Path
    Path(path).write_text('{}')
    with pytest.raises((ValueError, KeyError, RailsError)):
        adapter.apply(base,data,desc,[])


def test_rejects_wrong_parent_and_consumer_overlap(assets):
    data, desc, _, original, _, _ = assets
    base={'knowledge_stores':{'slot-contracts':{'store_role':'slot-contracts', 'snapshot':{'sha256':'f'*64}}}}
    with pytest.raises(ValueError, match='parent'):
        adapter.apply(base,data,desc,[])
    base['knowledge_stores']['slot-contracts']['snapshot']['sha256']=original['snapshot_sha256']
    with pytest.raises(ValueError, match='overlaps'):
        adapter.apply(base,data,desc,['slurp:train:new:other'])


def test_candidate_router_supplies_changed_card(assets):
    from speech_aware_evidence_acquisition.rails.knowledge_stores import build_knowledge_store_supply
    from test_candidate_semantic_supply import _program, _context, _artifact
    data, desc, _, original, _, _ = assets
    spec = {'store_role':'slot-contracts', 'snapshot':{'sha256':original['snapshot_sha256']}}
    base = {'knowledge_stores':{'slot-contracts':spec}}
    child = adapter.apply(base, data, desc, [])
    child['knowledge_stores']['slot-contracts']['candidate_supply'] = {'solve': {
        'provider':'slurp-candidate-semantic-v1',
        'rank_policy':'saea-ranked-candidate-round-robin-v1',
        'fuse_policy':'saea-candidate-semantic-passthrough-v1',
        'candidate_state_phase':'hypotheses', 'max_candidates':1,
        'max_values_per_candidate':2, 'max_total_values':2}}
    binding = load_knowledge_store_binding(child, data_root=data)
    router = build_knowledge_store_supply(binding, program=_program(), data_root=data,
                                           legal_candidate_ids=('datetime_query',))
    supply = router.automatic_for_phase('solve', _context(binding, _artifact('s1', ('datetime_query',))))
    ids = supply.fusion_decision.selected_value_ids
    assert 'slot-contracts::slot-contract:date' in ids
    blocks = KnowledgeValueRenderer(binding.composite_snapshot).assemble(ids)
    assert any('Retain the full date expression' in block['text'] for block in blocks)
