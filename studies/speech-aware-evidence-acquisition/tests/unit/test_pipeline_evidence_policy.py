import json
import pytest
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import candidates, parse_config, validate_evidence_refs


def inventory(*ids):
    return {'sample_id':'row-a','candidate_ids':[], 'sources':[{'source_id':v,'kind':'knowledge'} for v in ids]}


def test_alias_opt_in_maps_only_unique_supplied_source():
    rows=[{'evidence_refs':['intent:one']}]
    inv=inventory('store::intent:one')
    with pytest.raises(ValueError): validate_evidence_refs(rows,inv,[])
    assert validate_evidence_refs(rows,inv,[],policy='unique_source_alias_v1') == [{'raw':'intent:one','source_id':'store::intent:one'}]
    with pytest.raises(ValueError): validate_evidence_refs(rows,inventory('a::intent:one','b::intent:one'),[],policy='unique_source_alias_v1')
    with pytest.raises(ValueError): validate_evidence_refs(rows,inventory('store::other'),[],policy='unique_source_alias_v1')


def test_duplicate_policy_is_explicit_and_rejects_conflicting_content():
    row={'intent_id':'foo','distinction':'x','evidence_refs':[],'unresolved_fields':[]}
    text=json.dumps({'candidates':[row,row]})
    with pytest.raises(ValueError): candidates(text,3)
    assert len(candidates(text,3,duplicate_policy='identical_v1'))==1
    with pytest.raises(ValueError,match='conflicting'): candidates(json.dumps({'candidates':[row,{**row,'distinction':'y'}]}),3,duplicate_policy='identical_v1')
    with pytest.raises(ValueError,match='bound'): candidates(json.dumps({'candidates':[row]*4}),3,duplicate_policy='identical_v1')


def test_config_old_default_and_explicit_versions():
    old={'schema':'saea-intent-slot-pipeline-v1','slot_mode':'batched','max_candidates':3,'candidate_outside_recovery':False}
    assert parse_config(old)==old
    assert parse_config({**old,'evidence_policy':'unique_source_alias_v1'})['evidence_policy']=='unique_source_alias_v1'
    with pytest.raises(ValueError): parse_config({**old,'evidence_policy':'fuzzy'})


def test_raw_candidates_survive_unique_alias_validation():
    row={'intent_id':'foo','distinction':'x','evidence_refs':['intent:one'],'unresolved_fields':[]}
    rows=candidates(json.dumps({'candidates':[row]}),3)
    before=json.dumps(rows,sort_keys=True)
    assert validate_evidence_refs(rows,inventory('store::intent:one'),[],policy='unique_source_alias_v1')==[{'raw':'intent:one','source_id':'store::intent:one'}]
    assert json.dumps(rows,sort_keys=True)==before
