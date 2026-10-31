import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import select_real500_profile as m


def results():
    return [{'schema':'saea-real500-verified-result-v1','run_id':run,'profile':'profile-'+run,'rows':500,
        'official_finalized':True,'quality_identity_passed':True,'checkpoint_cost_reconciled':True,
        'generation_sealed':True,'generation_start_to_seal_seconds':seconds,'split_identity_hash':'same',
        'quality':{k:.72 for k in ('slu_f1','intent_accuracy','span_f1','scenario_accuracy')},
        'cost':{'calls_used':1000,'prompt_tokens_total':6000000,'completion_tokens_total':1600000},
        'selection_evidence':{'manifest':run}}
        for run,seconds in [('SAEA-SLURP-REAL500-TP4-S96-V1',1452),('SAEA-SLURP-REAL500-TP4-S128-V1',1383),('SAEA-SLURP-REAL500-TP8-S256-V1',2500)]]


def test_complete_throughput_choice_keeps_quality_and_holds384():
    r=results();d=m.choose(r)
    assert d['selected_run_id'].endswith('TP4-S128-V1')
    assert len(d['full_results'])==3 and not d['claims_system_gain_locked']
    assert d['candidates']==[x['selection_evidence'] for x in r]


@pytest.mark.parametrize('fault',['incomplete','missing','population','nan','quality','tp8-wins'])
def test_refuse_unfinished_comparison_or_pending384(fault):
    r=results()
    if fault=='incomplete':r[2]['official_finalized']=False
    elif fault=='missing':r.pop()
    elif fault=='population':r[2]['split_identity_hash']='other'
    elif fault=='nan':r[1]['generation_start_to_seal_seconds']=float('nan')
    elif fault=='quality':r[1]['quality']['slu_f1']=float('nan')
    else:r[2]['generation_start_to_seal_seconds']=1000
    with pytest.raises(ValueError):m.choose(r)
