import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import tfrl_current_distill as m
from test_tfrl_current_half_result import result


class Measure:
    name=m.common.MEASURE_VLLM
    calls=0
    def count(self,text):self.calls+=1;return len(text.split())
    def detail(self):return {'fixture':True}


def fixture():
    rows=[];trace={};ids=['fixture:'+str(i) for i in range(3)]
    results={a:result() for a in ('s1','s2','s3','s4','aa','lock','vote')}
    for a,r in results.items():
        good=a in ('s1','vote');template=result(good)['per_sample'][0]
        r['per_sample']=[dict(template,sample_id=sid) for sid in ids];r['aggregate']['samples']=3;r['aggregate']['official']['slu_f1']=float(good)
    for sid in ids:
        contacts=[]
        for arm in ('s1','s2','s3','s4','aa','lock'):
            rid=sid+'/'+arm
            payload={'sample_id':sid,'supplied_evidence':[{'evidence_id':'train::one',
                'content':'set alarm at six | alarm_set | time=six'}]}
            answer={'scenario':'alarm','action':'set','entities':[{'type':'time','filler':'six'}] if arm=='s1' else []}
            raw=json.dumps(answer);digest=hashlib.sha256(raw.encode()).hexdigest()
            trace[rid+'/request']={'payload':payload,'payload_sha256':'request'}
            trace[rid+'/response']={'text':raw,'response_text_sha256':digest}
            contacts.append({'request_id':rid,'request_sha256':'request','response_text_sha256':digest})
        rows.append({'sample_id':sid,'native_episode_receipt':{'role_visits':[{'stage':'solver','execution_receipt':{'contacts':contacts}}]}})
    plan={'row_plan':[{'row_id':sid} for sid in ids],'run_id':'run','attempt_id':'attempt','iteration':1,'half':'R01C-H1',
        'loop_id':m.common.LOOP_ID,'roster':{'file_sha256':'roster'}}
    bundle=SimpleNamespace(outputs_rows=rows,trace_records=trace,manifest={'raw_trace':{'sha256':'trace'}})
    tables={a:{r['sample_id']:r for r in result['per_sample']} for a,result in results.items()}
    credit=m.sampler.credit_rows(ids,1,4,m.sampler.scores_from_counts(tables),twin_arm='aa')
    return {'bundle':bundle,'plan':plan,'results':results,'credit':credit,'roster':{'split_identity_hash':'a'*64},'source':{'run_id':'run'}}


@pytest.mark.parametrize('fault',[None,'hash','row','evidence','visits','count','credit','estimate'])
def test_source_bound_distillation(fault,monkeypatch):
    loaded=fixture();measure=Measure()
    # Use the same canonical output renderer; fixture raw answers already canonical.
    monkeypatch.setattr(m.s3,'normalise_arm_answer',lambda raw,_:(raw,True))
    if fault=='hash':loaded['bundle'].trace_records['fixture:0/s1/response']['text']='different'
    elif fault=='row':loaded['bundle'].trace_records['fixture:0/s1/request']['payload']['sample_id']='other'
    elif fault=='evidence':loaded['bundle'].trace_records['fixture:0/s2/request']['payload']['supplied_evidence'][0]['content']='changed'
    elif fault=='visits':loaded['bundle'].outputs_rows[0]['native_episode_receipt']['role_visits']*=2
    elif fault=='count':loaded['bundle'].outputs_rows.pop()
    elif fault=='credit':loaded['credit'][0]['positive']=False
    elif fault=='estimate':measure.name=m.common.MEASURE_ESTIMATE
    if fault:
        with pytest.raises(ValueError):m.build(loaded,measure)
        assert measure.calls==0
    else:
        built=m.build(loaded,measure);receipt=built['receipt_core']
        assert receipt['consuming_half']=='R01C-H2' and receipt['counts']['rows_skipped']==0
        assert receipt['rule_bank']['rules']==1 and receipt['exemplar_bank']['entries']==1
        block=built['exemplar_bank']['by_scenario']['alarm']['block_text']
        assert '[evidence train::one] set alarm at six | alarm_set | time=six' in block
        assert len(built['provenance'])==18 and receipt['tokenizer_calls']>0
        assert not receipt['next_iteration_authorized']
