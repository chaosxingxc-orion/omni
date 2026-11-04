"""The output protocol restricts labels and envelope, never answer fillers."""
import copy
import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator
from speech_aware_evidence_acquisition.rails.slurp_solver_protocol import response_format, registered_response_format, FORMAT_ID

ROOT = Path(__file__).resolve().parents[2]


def inventory():
    return json.loads((ROOT/'configs/probes/solver-protocol/slurp-label-inventory.json').read_text())


def schema():
    inv = inventory()
    return response_format(inv['actions_by_scenario'], inv['entity_types'])['json_schema']['schema']


def test_all_sixty_original_pairs_admitted_without_cross_product():
    inv=inventory();s=schema();Draft202012Validator.check_schema(s);v=Draft202012Validator(s)
    pairs={(scenario,action) for scenario,actions in inv['actions_by_scenario'].items() for action in actions}
    assert len(pairs)==60 and len(inv['entity_types'])==55
    for scenario in inv['actions_by_scenario']:
        for action in {a for _,a in pairs}:
            answer={'scenario':scenario,'action':action,'entities':[]}
            assert v.is_valid(answer)==((scenario,action) in pairs)


def test_slot_combinations_and_open_fillers_are_not_restricted_by_training_frequency():
    v=Draft202012Validator(schema())
    for typ in inventory()['entity_types']:
        assert v.is_valid({'scenario':'audio','action':'volume_up','entities':[
            {'type':typ,'filler':'unseen literal 任意值'}, {'type':typ,'filler':'another occurrence'}]})


@pytest.mark.parametrize('value',[
    {'scenario':'music','action':'play_music','entities':[]},
    {'scenario':'play','action':'play_music','entities':[]},
    {'scenario':'play','action':'music'},
    {'scenario':'play','action':'music','entities':[{'type':'invented','filler':'x'}]},
    {'scenario':'play','action':'music','entities':[],'extra':True},
    [{'scenario':'play','action':'music','entities':[]}],
])
def test_invalid_envelopes_and_observed_illegal_pairs_refused(value):
    assert not Draft202012Validator(schema()).is_valid(value)


def test_builder_deterministic_and_does_not_mutate_inventory():
    inv=inventory();before=copy.deepcopy(inv)
    assert response_format(inv['actions_by_scenario'],inv['entity_types'])==response_format(dict(reversed(list(inv['actions_by_scenario'].items()))),list(reversed(inv['entity_types'])))
    assert inv==before
    with pytest.raises(ValueError):response_format({'play':['music','music']},['person'])
    with pytest.raises(ValueError):response_format({'play':'music'},['person'])


def test_registered_schema_uses_governed_ontology_without_fillers(monkeypatch):
    from speech_aware_evidence_acquisition.data import slurp
    from speech_aware_evidence_acquisition.rails import controllers
    from speech_aware_evidence_acquisition.core.base import RESPONSE_FORMATS
    inv=inventory()
    ontology=slurp.SlurpOntology(tuple(inv['actions_by_scenario']),(),{t:('secret filler',) for t in inv['entity_types']},inv['actions_by_scenario'])
    monkeypatch.setattr(slurp,'load_slurp_ontology',lambda lock,root:ontology)
    result=RESPONSE_FORMATS.resolve(FORMAT_ID)({},ROOT)
    assert result['json_schema']['schema']==schema()
    assert 'secret filler' not in json.dumps(result)
    with pytest.raises(ValueError):registered_response_format({},ROOT,query_id='forbidden')
