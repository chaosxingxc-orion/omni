import copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfrl_current_cost_audit as m


def fixture(monkeypatch):
    monkeypatch.setattr(m.s3,'normalise_arm_answer',lambda raw,normalizer:(raw,True))
    trace={};contacts=[];arms=[];events=[]
    for i,label in enumerate(('perception',*m.ARMS)):
        rid=f'request-{7-i}';cid='contact-'+label
        raw=json.dumps({'answer':'different' if label=='s4' else 'same'})
        digest=hashlib.sha256(raw.encode()).hexdigest()
        contacts.append({'agent_contact_id':cid,'request_id':rid,'request_sha256':'request-pin','response_text_sha256':digest})
        trace[rid+'/request']={'payload':{'sample_id':'row'},'payload_sha256':'request-pin'}
        trace[rid+'/response']={'text':raw,'response_text_sha256':digest}
        trace[rid+'/cost']={'calls':1,'prompt_tokens':10+i,'completion_tokens':20+i,
            'audio_seconds':1,'latency_seconds':30+i,'reasoning_tokens':5,'finish_reason':'stop','forced_close':False}
        events.append({'stage':'request_total','sample_id':'row','request_id':rid,'elapsed_seconds':31+i})
        if label!='perception':arms.append({'arm_id':label,'contact_id':cid})
    row={'sample_id':'row','supply_variants':{'arms':arms},'native_episode_receipt':{'role_visits':[
        {'role_id':'perception','execution_receipt':{'contacts':contacts[:1]}},
        {'role_id':'solver','execution_receipt':{'contacts':list(reversed(contacts[1:]))}}]}}
    return SimpleNamespace(outputs_rows=[row],trace_records=trace),[{'row_id':'row','reason':'positive'}],events


def test_costs_follow_custody_not_order_and_diversity_is_explicit(monkeypatch):
    bundle,credit,events=fixture(monkeypatch)
    result=m.summarize(bundle,credit,events)
    assert result['contacts']==7 and result['contacts_per_row_distribution']=={'7':1}
    assert result['groups']['s1']['prompt_tokens']==11
    assert result['groups']['lock']['prompt_tokens']==16
    assert result['per_contact_distributions']['s4']['request_total_seconds']['mean']==35
    assert result['answer_diversity']['normalized:k4_unique_sum']==2
    assert result['answer_diversity']['normalized:aa_equals_lock']==1
    assert result['cost_by_credit_disposition']['positive']['calls']==7
    assert result['missing_request_timing']==0


@pytest.mark.parametrize('fault',['response','request','arm','extra_cost','missing_cost','duplicate_timing','bad_cost','missing_credit'])
def test_audit_rejects_ambiguous_or_unbound_cost(monkeypatch,fault):
    bundle,credit,events=fixture(monkeypatch)
    if fault=='response':bundle.trace_records['request-1/response']['text']='{}'
    elif fault=='request':bundle.trace_records['request-1/request']['payload']['sample_id']='other'
    elif fault=='arm':bundle.outputs_rows[0]['supply_variants']['arms'][1]['contact_id']='unknown'
    elif fault=='extra_cost':bundle.trace_records['extra/cost']={}
    elif fault=='missing_cost':del bundle.trace_records['request-1/cost']
    elif fault=='duplicate_timing':events.append(events[0])
    elif fault=='bad_cost':bundle.trace_records['request-1/cost']['prompt_tokens']=float('nan')
    else:credit=[]
    with pytest.raises((ValueError,KeyError)):m.summarize(bundle,credit,events)


def test_missing_timing_is_counted_not_fabricated(monkeypatch):
    bundle,credit,events=fixture(monkeypatch)
    result=m.summarize(bundle,credit,[])
    assert result['missing_request_timing']==7
    assert 'request_total_seconds' not in result['per_contact_distributions']['s1']
