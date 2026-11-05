import copy
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).parents[2] / 'scripts/analysis/fold_paired_comparison.py'
spec = importlib.util.spec_from_file_location('fold_paired_comparison', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

def inputs():
    items=[{'sample_id':f'slurp:train:{i:06d}', 'utterance_id':str(i)} for i in range(1080)]
    rows=[{'sample_id':x['sample_id'],'scenario_action_exact':1,'slot_f1_exact_fp':0,
           'slot_f1_exact_fn':0,'slu_f1_word_tp':1,'slu_f1_char_tp':1,
           'slu_f1_word_fp':0,'slu_f1_char_fp':0,'slu_f1_word_fn':0,'slu_f1_char_fn':0} for x in items]
    aggregate={'samples':1080,'identity_check':{'passed':True},'official':{'slu_f1':1.0},
        'evaluator_commit':'abc','scorer_version':'v1','split_id':'train',
        'projection':{'applied':False},'filler_normalization':{'applied':False}}
    entry={'aggregate':aggregate,'per_sample':rows}
    return {'items':items},entry,copy.deepcopy(entry)

def test_full_roster_and_strict_respects_slot_errors():
    roster,left,right=inputs();right['per_sample'][0]['slot_f1_exact_fp']=1
    ids,identity,tables,strict=module.validate_tables(roster,[left,right])
    assert len(ids)==1080 and strict[0][ids[0]]['strict']==1 and strict[1][ids[0]]['strict']==0

def test_missing_duplicate_and_normalizer_mismatch_rejected():
    roster,left,right=inputs();right['per_sample'].pop()
    with pytest.raises(ValueError):module.validate_tables(roster,[left,right])
    roster,left,right=inputs();right['per_sample'][0]['sample_id']=right['per_sample'][1]['sample_id']
    with pytest.raises(ValueError):module.validate_tables(roster,[left,right])
    roster,left,right=inputs();right['aggregate']['filler_normalization']['applied']=True
    with pytest.raises(ValueError,match='normalizer'):module.validate_tables(roster,[left,right])

def test_missing_roster_is_not_reported_as_score(tmp_path):
    import json
    cfg={'schema':module.SCHEMA,'repo':str(tmp_path),'data':str(tmp_path),'roster':'missing.json',
         'roster_sha256':'0'*64,'split_identity_hash':'x','arms':{'control':{},'split':{}},'seed':1,'draws':10,'comparison_spec':'test specification'}
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError):module.run(p,tmp_path/'out')
    assert not (tmp_path/'out').exists()
