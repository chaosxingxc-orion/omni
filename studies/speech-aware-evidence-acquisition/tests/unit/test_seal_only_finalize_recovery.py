import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

p=Path(__file__).parents[2]/'scripts/analysis/recover_seal_only_finalize.py'
s=importlib.util.spec_from_file_location('seal_finalize_recovery',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def evidence():
    cfg={'run_id':'run','attempt_id':'successor','rows':2,'checkpoint_sha256':'a'*64}
    manifest={'status':'COMPLETED','scores':None,'run_id':'run','attempt':{'attempt_id':'successor'},'study_commit':'c'*40,
              'cost':{'calls_used':0},'config':{'config_hash':'d'*64},'protocol_hash':'e'*64,
              'predecessor_attempt':{'attempt_id':'old','checkpoint_sha256':'a'*64,'counts':{'kept':2,'flown':0,'partial':0}}}
    bundle=SimpleNamespace(attempt_state='OPENED',manifest=manifest,outputs_rows=[{},{}],split_ids={'a','b'})
    pre={'config_hash':'d'*64,'protocol_hash':'e'*64,'sample_count':2,'resume':{'seal_only':True,
        'counts':{'roster':2,'kept':2,'flown':0,'partial':0,'unstarted':0},'worst_case_contacts':0,'predecessor_contacts':5}}
    receipts={'stages':{'preflight':pre,'preprocess':{},'serve':{},'run_phases':{'seal_only':True,'resume':{'kept':2,'flown':0}}}}
    gen={'schema':'saea-seal-only-generation-manifest-v1','run_id':'run','attempt_id':'successor','seal_only':True,
         'cost_provenance':{'predecessor_attempt_ids':['old']},'cost':{'calls_used':5}}
    return cfg,bundle,receipts,gen,'finalize_bundle score_bundle delivery config differs from recorded run',{'generation_commit':'c'*40}


def test_exact_seal_only_evidence_passes():
    m.validate_evidence(*evidence())


@pytest.mark.parametrize('fault',['opened','score','roster','partial','flown','checkpoint','cost','failure','config','stages'])
def test_changed_source_or_failure_is_refused(fault):
    args=list(evidence());cfg,bundle,receipts,gen,failed,identity=args
    if fault=='opened':bundle.attempt_state='FINALIZED'
    elif fault=='score':bundle.manifest['scores']={'path':'existing'}
    elif fault=='roster':bundle.outputs_rows=[]
    elif fault=='partial':receipts['stages']['preflight']['resume']['counts']['partial']=1
    elif fault=='flown':bundle.manifest['predecessor_attempt']['counts']['flown']=1
    elif fault=='checkpoint':bundle.manifest['predecessor_attempt']['checkpoint_sha256']='b'*64
    elif fault=='cost':gen['cost']['calls_used']=0
    elif fault=='failure':args[4]='different error'
    elif fault=='config':receipts['stages']['preflight']['config_hash']='bad'
    else:receipts['stages']['seal']={}
    with pytest.raises(ValueError):m.validate_evidence(*args)
