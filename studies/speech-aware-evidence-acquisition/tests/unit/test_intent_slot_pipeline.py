import pytest
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import candidates, canonical, slot_results, decision

@pytest.mark.parametrize('rows', [[], [{'intent_id': 'a'}, {'intent_id': 'a'}], [{'intent_id': 'illegal'}]])
def test_invalid_candidates(rows):
    with pytest.raises(ValueError):
        candidates(canonical({'candidates': rows}), 3, {'a', 'b'})

def test_cross_candidate_and_missing_results_refused():
    row = {'candidate_id': 'b', 'entities': [], 'evidence_refs': [], 'unresolved_fields': []}
    with pytest.raises(ValueError):
        slot_results(canonical({'candidates': [row]}), ['a'])
    with pytest.raises(ValueError):
        slot_results(canonical({'candidates': [row]}), ['a', 'b'])
    assert slot_results(canonical({'candidates': [row]}), ['b'])['b']['entities'] == []

def test_decision_cannot_cross_candidate_or_change_entities():
    rows = [{'candidate_id': 'a'}]
    slots = {'a': {'entities': [{'type': 'person', 'filler': 'Alice'}]}}
    with pytest.raises(ValueError):
        decision('{"intent":"b","entities":[]}', rows, slots, lambda row: row['intent'])
    with pytest.raises(ValueError):
        decision('{"intent":"a","entities":[]}', rows, slots, lambda row: row['intent'])


def test_preparation_reuses_roster_and_adds_real_slot_supply():
    import json
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'scripts/analysis'))
    from derive_lock_only_baseline import derive as lock
    from prepare_intent_slot_candidate import derive
    source=json.loads((root/'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json').read_text())
    baseline=lock(source)
    prepared=derive(baseline)
    from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    assert len(load_phase_material_specs(prepared,default_phase_program(prepared['policy'])))==5
    assert prepared['knowledge_stores']['slot-contracts']['candidate_supply']['slot']==baseline['knowledge_stores']['slot-contracts']['candidate_supply']['solver']
    assert prepared['knowledge_stores']['intent-boundaries']['phase_supply']['perception']==baseline['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    assert prepared['native_episode']['intent_slot_pipeline']['candidate_outside_recovery'] is False
    assert 'slot' not in baseline['native_episode']['roles']


def test_slot_response_format_uses_actual_ontology_pairs(monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.data import slurp
    from speech_aware_evidence_acquisition.rails.adapters.slurp_intent_slot import registered_slot_format
    monkeypatch.setattr(slurp,'load_slurp_ontology',lambda lock,root:SimpleNamespace(
        intents=('play_music','calendar_create'),entity_types={'person':(), 'date':()}))
    value=registered_slot_format({},None)['json_schema']['schema']['properties']['candidates']['items']['properties']
    assert value['candidate_id']['enum']==['play_music','calendar_create']
    assert set(value['entities']['items']['properties']['type']['enum'])=={'person','date'}

@pytest.mark.parametrize('mode', ['batched', 'per_candidate'])
@pytest.mark.parametrize('maximum', [1, 2, 3])
def test_recovery_admission_derives_actual_graph_without_carrier_whitelist(mode, maximum):
    from test_native_episode_config import config
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    values = config()
    values.update(transport='vllm', arm='raw-speech-agent-native-tools', carrier_lock_key='another-carrier',
                  policy='phase-native-intent-slot-'+mode.replace('_','-')+'-v1')
    native = values['native_episode']
    native.update(review_loop='off', finite_completion_policy='deliver-latest-or-abstain',
                  intent_slot_pipeline={'schema':'saea-intent-slot-pipeline-v1','slot_mode':mode,
                                        'max_candidates':maximum,'candidate_outside_recovery':False})
    for name in ('reviewer','planner'): native['roles'][name]['max_iterations']=0
    native['roles']['slot']={'phase_id':'slot','max_iterations':1}
    assert native_contact_custody_enabled(values)
    native['roles']['slot']['max_iterations']=2
    assert not native_contact_custody_enabled(values)
    native['roles']['slot']['max_iterations']=1
    native['intent_slot_pipeline']['slot_mode']='per_candidate' if mode=='batched' else 'batched'
    assert not native_contact_custody_enabled(values)


def test_evidence_refs_are_supplied_and_branch_scoped():
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import evidence_inventory, validate_evidence_refs
    inventory=evidence_inventory(sample_id='one',speech_packet_id='audio-one',speech_sha256='a'*64,
        messages=[{'content':[{'evidence_id':'knowledge-one','source_uri':'snapshot://one','text':'meaning'}]}],candidate_ids=['a'])
    validate_evidence_refs([{'evidence_refs':['audio-one','knowledge-one']}],inventory,['a'])
    validate_evidence_refs([{'evidence_refs':[]}],inventory,['a'])
    with pytest.raises(ValueError): validate_evidence_refs([{'evidence_refs':['audio-two']}],inventory,['a'])
    with pytest.raises(ValueError): validate_evidence_refs([{'evidence_refs':['knowledge-one']}],inventory,['b'])


@pytest.mark.parametrize('maximum', [1, 2, 3])
def test_continuous_dispatch_uses_real_native_budget(maximum):
    from types import SimpleNamespace
    from pathlib import Path
    from test_native_episode_config import config
    from test_rails_phase_runner import _FakeAdapter, _bindings
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config, NativeEpisodeRunBinding
    from speech_aware_evidence_acquisition.rails.phase_runner import run_continuous_program
    values=config();native=values['native_episode']
    native.update(review_loop='off',finite_completion_policy='deliver-latest-or-abstain',
        intent_slot_pipeline={'schema':'saea-intent-slot-pipeline-v1','slot_mode':'per_candidate',
                              'max_candidates':maximum,'candidate_outside_recovery':False})
    for name in ('reviewer','planner'):native['roles'][name]['max_iterations']=0
    native['roles']['slot']={'phase_id':'slot','max_iterations':1}
    program=default_phase_program('phase-native-intent-slot-per-candidate-v1')
    parsed=load_native_episode_config(values,program)
    material=lambda *args:None
    binding=NativeEpisodeRunBinding(config=parsed,program=program,material_for_phase=material,
                                   render_request=lambda x:x,request_gate=lambda x:None)
    captured={}
    def runner(**kw):captured.update(kw);return SimpleNamespace(results=(SimpleNamespace(sample_id='s1'),))
    run_continuous_program(program=program,adapter=_FakeAdapter(),plan=SimpleNamespace(),
        data_root=Path('.'),core_client=SimpleNamespace(),split_id='all',knowledge_snapshot=SimpleNamespace(),
        speech_packet_bindings=_bindings(('s1',)),trace_sink=SimpleNamespace(),decoding_params={},
        material_for_phase=material,native_episode_binding=binding,agent_runner=runner)
    assert captured['max_contacts']==maximum+2
