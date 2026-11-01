"""Historical input integrity and accounting include failed observed calls."""
import copy
import json
from dataclasses import asdict
import pytest
from test_frozen_core_adapter import _plan
from test_tfgrpo_learning import group,MODEL
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal,canonical,digest
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha,render,LearningError
from speech_aware_evidence_acquisition.rails.tfgrpo_projection import summary_trajectory
from speech_aware_evidence_acquisition.rails.tfgrpo_summary_import import load_import


def fixture(root):
    plan=_plan();groups=[group()];g=groups[0];r=g['rollouts'][0]
    common={'agent_objective':'Intent','learning_objective':'Evidence','num_experiences':2}
    batch={'model_id':MODEL,'groups':groups,'common':common,'learning_config_sha256':'a'*64}
    def write(rel,doc):
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical(doc))
        return {'relpath':rel,'sha256':digest(p.read_bytes())}
    messages=[{'role':'system','content':render('SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP',**common)},
        {'role':'user','content':render('SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP',question=g['query'],answer=g['answer'],critique=r['critique'],trajectory=summary_trajectory(g['query'],r['trajectory']))}]
    request={'stage':'summary','identity':r['trajectory_id'],'messages':messages,'batch_sha256':sha(batch),'learning_config_sha256':'a'*64,'model_id':MODEL}
    journal=ContactJournal(root,plan,'prepared-context-text')
    payload={'prepared_text':'saea-prepared-text-v1','request_id':'one','sample_id':g['query_id'],'history':[{'role':'agent_state','content':json.dumps({'stage':'summary','source_messages':messages})},{'role':'control','contact_id':'summary-one'}]}
    journal.intent(payload);body={'text':'Useful summary','finish_reason':'stop','usage':{'prompt_tokens':10,'completion_tokens':4}}
    journal.response('one',sha(payload),canonical(body),1)
    second={**payload,'request_id':'two'};journal.intent(second)
    journal.response('two',sha(second),canonical({'text':None,'finish_reason':'length','usage':{'prompt_tokens':12,'completion_tokens':4096}}),1)
    bridge=f'runs/{plan.run_id}/{plan.attempt_id}/tfgrpo/contacts';key=sha({'stage':'summary','identity':r['trajectory_id']})
    intent=write(bridge+'/'+key+'.intent.json',{'request':request,'payload':{**payload,'request_id':'prepared-id'}})
    response=write(bridge+'/'+key+'.response.json',{'request_sha256':sha(request),'response':{'receipt_id':f'{plan.run_id}/{plan.attempt_id}/one','model_id':MODEL,'text':body['text'],'usage':{'calls':1,**body['usage']},'response_sha256':digest(canonical(body)),'request_sha256':sha(payload)}})
    attempt={'state':'ABORTED','contact_kind':'prepared-text','run_id':plan.run_id,'attempt_id':plan.attempt_id,'carrier_lock_key':plan.carrier_lock_key,'reserved':{'calls':plan.planned_model_calls,'audio_seconds':float(plan.planned_speech_audio_seconds)},'actual':{'calls':2,'audio_seconds':0.0}}
    doc={'schema':'saea-tfgrpo-summary-import-v1','source_plan':asdict(plan),'source_attempt':write(f'attempts/{plan.run_id}/{plan.attempt_id}.json',attempt),'source_batch':write('source.json',{'batch':batch}),'source_bridge':bridge,'records':[{'trajectory_id':r['trajectory_id'],'intent':intent,'response':response}]}
    return write('import.json',doc),dict(data_root=root,groups=groups,common=common,model_id=MODEL),journal,doc,write


def test_import_preserves_all_original_spend_and_transport_identity(tmp_path):
    pin,args,j,doc,write=fixture(tmp_path)
    summaries,custody=load_import(pin,**args)
    assert summaries=={'q1-t0':'Useful summary'}
    assert custody['original_total_usage']=={'calls':2,'prompt_tokens':22,'completion_tokens':4100}
    assert custody['imported_summaries']==1


@pytest.mark.parametrize('fault',['unknown','orphan','pin','duplicate','prompt','payload','cost','source'])
def test_changed_or_incomplete_history_cannot_be_imported(tmp_path,fault):
    pin,args,j,doc,write=fixture(tmp_path)
    if fault=='unknown':j._path('two','response').unlink()
    elif fault=='orphan':j._path('two','intent').unlink()
    elif fault=='pin':(tmp_path/doc['records'][0]['response']['relpath']).write_text('{}')
    elif fault=='duplicate':doc['records']*=2
    elif fault in ('prompt','payload'):
        row=doc['records'][0];p=tmp_path/row['intent']['relpath'];v=json.loads(p.read_bytes())
        if fault=='prompt':v['request']['messages'][1]['content']='changed'
        else:v['payload']['history'][1]['contact_id']='different'
        row['intent']=write(row['intent']['relpath'],v)
    elif fault=='cost':
        v=json.loads((tmp_path/doc['source_attempt']['relpath']).read_bytes());v['actual']['calls']=1
        doc['source_attempt']=write(doc['source_attempt']['relpath'],v)
    else:args['groups']=copy.deepcopy(args['groups']);args['groups'][0]['answer']='new answer'
    pin=write('import.json',doc)
    with pytest.raises((LearningError,ValueError)):load_import(pin,**args)
