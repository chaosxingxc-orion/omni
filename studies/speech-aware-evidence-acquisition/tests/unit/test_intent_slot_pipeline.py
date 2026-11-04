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
