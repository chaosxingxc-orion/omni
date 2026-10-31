import copy,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/flights/nx'),str(ROOT/'scripts/analysis')]
import fold_a_c2_comparator as m
from derive_lock_only_baseline import derive as lock_only


def source():
    b=lock_only(m.read(ROOT/'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json'),repo=ROOT)
    b['native_contact_fit']={'run_id':'old','attempt_id':'old'}
    e={'cells':[{'model_fragment':'original'}],'service_session':'original','stage3_measurement':{'rows':500}}
    pins={p:str(i)*64 for i,p in enumerate((m.inputs.SEGMENTS,m.inputs.CONTENT,m.inputs.RECEIPT,
        m.inputs.ADOPTION,m.inputs.ALIAS_RECEIPT,m.inputs.PROVENANCE),1)}
    return b,e,pins


def test_only_query_and_agreement_change_preserving_c2_and_original_pool():
    b,e,p=source();before=copy.deepcopy((b,e));d,x=m.derive(ROOT,b,e,p)
    assert (b,e)==before
    assert d['policy']==b['policy'] and d['native_episode']['consumption_module']['lock']=='v2.2-npu-port'
    for k in ('roles','prompt_contact_limits','deliver_normaliser','max_revisions'):
        assert d['native_episode'][k]==b['native_episode'][k]
    for k in ('release','snapshot'):assert d['knowledge_stores']['train-exemplars'][k]==b['knowledge_stores']['train-exemplars'][k]
    old=b['knowledge_stores']['train-exemplars']['phase_supply']['solver'];new=d['knowledge_stores']['train-exemplars']['phase_supply']['solver']
    assert {k:v for k,v in old.items() if k!='query_source'}=={k:v for k,v in new.items() if k!='query_source'}
    assert new['query_source']['tool_contact_ledger_row']==m.inputs.RUN
    assert d['knowledge_stores']['slot-contracts']==b['knowledge_stores']['slot-contracts']
    assert x['cells'][0]['allowed_execution_stages']==['check'] and x['cells'][0]['readiness']['state']=='held'
    assert x['cells'][0]['model_fragment']=='original' and x['service_session']=='original'
    assert len(x['populations'][0]['ids'])==1080
    assert not any('fold-b' in p['relpath'] or 'FOLD-B' in p['relpath'] for p in d['preparation_source_receipts'])


@pytest.mark.parametrize('fault',['pin','loop','policy','recall'])
def test_invalid_inputs_refused(fault):
    b,e,p=source()
    if fault=='pin':p[m.inputs.CONTENT]='0'*64
    elif fault=='loop':b['native_episode']['review_loop']='on'
    elif fault=='policy':b['policy']='phase-native-module-v1-c17'
    else:b['knowledge_stores']['train-exemplars']['phase_supply']['solver']['k']=1
    with pytest.raises((ValueError,m.cm.ConsumptionModuleConfigError)):m.derive(ROOT,b,e,p)


@pytest.mark.parametrize('fault',[None,'recording','hidden','target','facet','duplicate'])
def test_pool_exclusion_checks_utterance_not_recording(fault):
    ids=[f'slurp:train:{i}:headset' for i in range(1080)];sid='slurp:train:9000:farfield'
    s=NS(source_sample_ids=[sid],facets=[NS(facet_id='f',source_sample_ids=[sid])],
        values=[NS(source_sample_ids=[sid],associated_facet_ids=['f'],task_target='intent')])
    if fault=='recording':s.source_sample_ids.append('slurp:train:0:farfield')
    elif fault=='hidden':s.values[0].source_sample_ids.append(ids[0])
    elif fault=='target':s.values[0].task_target=None
    elif fault=='facet':s.values[0].associated_facet_ids=['other']
    elif fault=='duplicate':ids[-1]=ids[0]
    if fault:
        with pytest.raises(ValueError):m.verify_membership(ids,s)
    else:assert m.verify_membership(ids,s)['utterance_overlap']==0
