"""Model-driven update, durable resume and fixed-policy provenance behavior."""
import copy
import json
import multiprocessing
import os
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, LearningError, UnknownLearningContact, empty_experience,
    experience_context, parse_operations, apply_operations, render,
)

MODEL='frozen-core-digest'; POLICY='1'*64; LEARNING_CONFIG='a'*64


def group(parent=None, qid='q1', rewards=(0,1)):
    parent=parent or empty_experience(MODEL)
    return {'query_id':qid,'query':'Example spoken request','answer':'Training-only reference',
        'rollouts':[{'trajectory_id':f'{qid}-t{i}','trajectory':f'Trajectory {i}',
                     'critique':f'External task reward {r}','reward':r,'model_id':MODEL,
                     'policy_sha256':POLICY,'experience_sha256':parent['sha256']}
                    for i,r in enumerate(rewards)]}


class Model:
    def __init__(self, operations=None):
        self.requests=[]
        self.operations=operations or [{'operation':'ADD','id':None,'content':'Conditional rule: distinguish action from object.'}]
    def __call__(self, request):
        self.requests.append(request)
        stage=request['stage']
        if stage=='summary':text='Observed action and reward summary.'
        elif stage=='advantage':text='<Experiences>Use action evidence to resolve intent.</Experiences>'
        else:text=json.dumps(self.operations)
        return {'model_id':MODEL,'text':text,'receipt_id':str(len(self.requests))+request['identity'],
                'usage':{'calls':1,'prompt_tokens':10,'completion_tokens':5}}


def learn(root, model, groups=None, parent=None, **options):
    parent=parent or empty_experience(MODEL)
    config=options.pop('learning_config_sha256',LEARNING_CONFIG)
    learner=ExperienceLearner(root,model_id=MODEL,metered_call=model,learning_config_sha256=config,max_calls=20,**options)
    result=learner.learn(groups or [group(parent)],parent,agent_objective='Intent and slot prediction',
                         learning_objective='Generalizable evidence use',policy_sha256=POLICY)
    return learner,result


def test_full_model_driven_chain_and_next_iteration_uses_model_proposed_content(tmp_path):
    model=Model();learner,result=learn(tmp_path/'first',model)
    assert [r['stage'] for r in model.requests]==['summary','summary','advantage','group_update','batch_update']
    assert list(result['items'].values())==[model.operations[0]['content']]
    assert result['learning_usage']=={'calls':5,'prompt_tokens':50,'completion_tokens':25}
    assert learner.last_invocation=={'new_calls':5,'reused_calls':0}
    assert not result['deployment_accepted']
    assert model.operations[0]['content'] in experience_context(result,'q2')
    key=next(iter(result['items']))
    model2=Model([{'operation':'UPDATE','id':key,'content':'Refined rule: require an explicit action cue.'}])
    _,second=learn(tmp_path/'second',model2,[group(result,'q2')],result)
    assert second['items'][key]==model2.operations[0]['content']
    assert any(model.operations[0]['content'] in m['content'] for r in model2.requests if r['stage']=='group_update' for m in r['messages'])
    with pytest.raises(LearningError,match='held-out'):experience_context(result,'q1')


def test_finished_resume_uses_no_model_and_detects_evidence_tampering(tmp_path):
    model=Model();_,first=learn(tmp_path,model)
    resumed,result=learn(tmp_path,lambda _:pytest.fail('must not call'))
    assert result==first and resumed.last_invocation['new_calls']==0
    path=next((tmp_path/'contacts').glob('*.response.json'));path.write_text('{}')
    with pytest.raises(LearningError,match='evidence changed'):learn(tmp_path,model)


def test_real_process_exit_after_observed_summary_reuses_response(tmp_path):
    # The next contact's intent is never created: process stops after the first
    # observed response publication, before the next network operation.
    from speech_aware_evidence_acquisition.rails import tfgrpo_learning as module
    root=tmp_path/'batch'
    def child():
        original=module.publish_new
        def publish(path,blob):
            original(path,blob)
            if str(path).endswith('.response.json'):os._exit(73)
        module.publish_new=publish
        learn(root,Model(),concurrency=1)
    process=multiprocessing.get_context('fork').Process(target=child)
    process.start();process.join(20)
    if process.is_alive():process.kill();process.join();pytest.fail('crash boundary not reached')
    assert process.exitcode==73
    model=Model();learner,result=learn(root,model,concurrency=1)
    assert len(model.requests)==4 and learner.last_invocation=={'new_calls':4,'reused_calls':1}
    assert result['learning_usage']['calls']==5


def test_unknown_is_not_retried_even_in_new_learner(tmp_path):
    count=[]
    def timeout(request):count.append(request);raise TimeoutError('unknown')
    with pytest.raises(TimeoutError):learn(tmp_path,timeout,concurrency=1)
    before=len(count)
    with pytest.raises(UnknownLearningContact):learn(tmp_path,timeout,concurrency=1)
    assert len(count)==before


@pytest.mark.parametrize('rewards',[(0,0),(1,1)])
def test_tied_binary_groups_do_not_invoke_model(tmp_path,rewards):
    _,result=learn(tmp_path,lambda _:pytest.fail('tie must not learn'),[group(rewards=rewards)])
    assert result['mixed_groups']==0 and result['learning_usage']['calls']==0 and result['items']=={}


@pytest.mark.parametrize('field,value',[('reward',.5),('reward',True),('model_id','different'),('policy_sha256','2'*64),('experience_sha256','3'*64)])
def test_invalid_rollout_provenance_refused_before_calls(tmp_path,field,value):
    g=group();g['rollouts'][0][field]=value
    with pytest.raises(LearningError):learn(tmp_path,lambda _:pytest.fail('must refuse'),[g])


def test_insufficient_complete_budget_refuses_before_model(tmp_path):
    learner=ExperienceLearner(tmp_path,model_id=MODEL,learning_config_sha256=LEARNING_CONFIG,metered_call=lambda _:pytest.fail('must refuse'),max_calls=4)
    with pytest.raises(LearningError,match='budget'):
        learner.learn([group()],empty_experience(MODEL),agent_objective='task',learning_objective='goal',policy_sha256=POLICY)


def test_invalid_model_operation_is_observed_and_never_silently_added(tmp_path):
    model=Model([{'operation':'UPDATE','id':'missing','content':'bad'}])
    with pytest.raises(LearningError,match='unknown'):learn(tmp_path,model)
    assert not (tmp_path/'candidate.json').exists()
    calls=len(model.requests)
    with pytest.raises(LearningError,match='unknown'):learn(tmp_path,model)
    assert len(model.requests)==calls


def test_delete_keep_and_conflicting_ops():
    items={'E1':'old','E2':'retain'}
    ops=parse_operations(json.dumps([{'operation':'DELETE','id':'E1','content':''},
                                    {'operation':'NONE','id':None,'content':''}]),items,max_operations=4)
    assert apply_operations(items,ops,max_items=4,max_characters=100)=={'E2':'retain'}
    with pytest.raises(LearningError):parse_operations(json.dumps([ops[0],ops[0]]),items,max_operations=4)
    with pytest.raises(LearningError):apply_operations(items,[{'operation':'ADD','id':None,'content':'long'}],max_items=2,max_characters=100)


def test_template_tokens_in_input_are_not_interpolated_again():
    output=render('SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP',question='{{ answer }}',answer='secret',critique='ok',trajectory='x')
    assert '{{ answer }}' in output


def test_changed_batch_identity_cannot_reuse_journal(tmp_path):
    learn(tmp_path,Model())
    g=group();g['query']='changed'
    with pytest.raises(LearningError,match='binding changed'):learn(tmp_path,Model(),[g])


def test_failed_batch_usage_includes_known_calls_and_unknown_reservation(tmp_path):
    model=Model()
    def call(request):
        if request['stage']=='advantage':raise TimeoutError('unknown advantage')
        return model(request)
    learner=ExperienceLearner(tmp_path,model_id=MODEL,learning_config_sha256=LEARNING_CONFIG,metered_call=call,max_calls=10)
    with pytest.raises(TimeoutError):
        learner.learn([group()],empty_experience(MODEL),agent_objective='task',learning_objective='goal',policy_sha256=POLICY)
    assert learner.accounting()=={'attempted_contacts':3,'observed_contacts':2,'unknown_contacts':1,
        'known_usage':{'calls':2,'prompt_tokens':20,'completion_tokens':10},
        'unknown_spend_excluded_from_known_usage':True}


def test_same_model_and_metered_usage_are_checked_before_update(tmp_path):
    model=Model()
    def wrong(request):
        result=model(request);result['model_id']='other';return result
    with pytest.raises(LearningError,match='another model'):learn(tmp_path,wrong,concurrency=1)
    assert not (tmp_path/'candidate.json').exists()


def test_self_conditioning_and_unsupported_shared_policy_are_refused(tmp_path):
    _,parent=learn(tmp_path/'first',Model())
    assert experience_context(parent,'q1',purpose='learning')
    key=next(iter(parent['items']))
    _,second=learn(tmp_path/'second',Model([{'operation':'UPDATE','id':key,'content':'Improved model rule.'}]),[group(parent)],parent)
    assert second['learning_query_reuse']==['q1']
    with pytest.raises(LearningError,match='held-out'):experience_context(second,'q1')
    learner=ExperienceLearner(tmp_path/'third',model_id=MODEL,learning_config_sha256=LEARNING_CONFIG,metered_call=Model(),max_calls=10)
    result=learner.learn([group()],empty_experience(MODEL),agent_objective='task',learning_objective='goal',
                         rollout_kind='shared-perception-adaptation',policy_sha256=POLICY)
    assert result['rollout_kind']=='shared-perception-adaptation'


def test_learning_configuration_is_pinned_in_request_and_resume(tmp_path):
    model=Model();learn(tmp_path,model)
    assert {r['learning_config_sha256'] for r in model.requests}=={LEARNING_CONFIG}
    with pytest.raises(LearningError,match='binding changed'):
        learn(tmp_path,lambda _:pytest.fail('must refuse changed configuration'),learning_config_sha256='b'*64)


def test_preflight_rejection_does_not_create_unknown_intent(tmp_path):
    def reject(request):raise ValueError('deterministic context rejection')
    with pytest.raises(ValueError,match='context rejection'):
        learn(tmp_path,lambda _:pytest.fail('must not send'),preflight=reject,concurrency=1)
    assert not list((tmp_path/'contacts').glob('*.intent.json'))
    learner,result=learn(tmp_path,Model(),preflight=lambda _:None,concurrency=1)
    assert learner.accounting()['unknown_contacts']==0 and result['learning_usage']['calls']==5


def test_group_pipelines_overlap_bounded_and_merge_in_input_order(tmp_path):
    import threading
    barrier=threading.Barrier(2)
    lock=threading.Lock();state={'active':0,'maximum':0};seen=[]
    model=Model()
    def call(request):
        if request['stage']=='advantage':
            with lock:
                state['active']+=1;state['maximum']=max(state['maximum'],state['active'])
            if request['identity'] in ('q1','q2'):barrier.wait(timeout=5)
        with lock:
            response=model(request)
            if request['stage']=='group_update':
                response['text']=json.dumps([{'operation':'ADD','id':None,'content':request['identity']}])
                state['active']-=1
            if request['stage']=='batch_update':seen.append(request)
        return response
    learn(tmp_path,call,[group(qid=f'q{i}') for i in range(1,4)],concurrency=2)
    assert state=={'active':0,'maximum':2}
    content=seen[0]['messages'][1]['content']
    assert content.index('"content": "q1"')<content.index('"content": "q2"')<content.index('"content": "q3"')


def test_real_process_exit_after_runtime_receipt_reconciles_without_resend(tmp_path):
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical,sha
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    root=tmp_path/'batch';runtime=tmp_path/'runtime';runtime.mkdir()
    def child():
        def call(request):
            response=Model()(request)
            publish_new(runtime/(sha(request)+'.json'),canonical({
                'request_sha256':sha(request),'response':response,'response_sha256':sha(response)}))
            os._exit(74)
        learn(root,call,concurrency=1)
    process=multiprocessing.get_context('fork').Process(target=child)
    process.start();process.join(20)
    if process.is_alive():process.kill();process.join();pytest.fail('runtime crash boundary not reached')
    assert process.exitcode==74
    assert len(list((root/'contacts').glob('*.intent.json')))==1
    assert not list((root/'contacts').glob('*.response.json'))
    checked=[]
    def reconcile(request):
        checked.append(request)
        p=runtime/(sha(request)+'.json')
        return json.loads(p.read_bytes()) if p.exists() else None
    model=Model();learner,result=learn(root,model,concurrency=1,reconcile=reconcile)
    assert len(checked)==1 and len(model.requests)==4
    assert learner.last_invocation=={'new_calls':4,'reused_calls':1}
    assert result['learning_usage']['calls']==5


@pytest.mark.parametrize('corruption',['request','response','receipt','model'])
def test_reconciliation_refuses_wrong_receipt_without_send(tmp_path,corruption):
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    def timeout(request):raise TimeoutError('unknown')
    with pytest.raises(TimeoutError):learn(tmp_path,timeout,concurrency=1)
    def reconcile(request):
        response=Model()(request)
        if corruption=='receipt':response['receipt_id']=''
        if corruption=='model':response['model_id']='other'
        return {'request_sha256':'b'*64 if corruption=='request' else sha(request),
                'response':response,'response_sha256':'c'*64 if corruption=='response' else sha(response)}
    with pytest.raises(LearningError):
        learn(tmp_path,lambda _:pytest.fail('unknown must not send'),concurrency=1,reconcile=reconcile)
    assert not list((tmp_path/'contacts').glob('*.response.json'))


def test_reconciliation_without_receipt_preserves_unknown(tmp_path):
    def timeout(request):raise TimeoutError('unknown')
    with pytest.raises(TimeoutError):learn(tmp_path,timeout,concurrency=1)
    with pytest.raises(UnknownLearningContact):
        learn(tmp_path,lambda _:pytest.fail('must never send'),concurrency=1,reconcile=lambda _:None)


@pytest.mark.parametrize('text',[
    '<Experiences>1. Use actual evidence.\n2. Check intent boundaries.</Experiences>',
    '<Learning Objective Focus>Analysis\n<Experiences>\n1. Use actual evidence.\n2. Check intent boundaries.',
])
def test_advantage_accepts_unambiguous_final_heading_without_changing_content(text):
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import semantic_advantage
    assert semantic_advantage(text)=='1. Use actual evidence.\n2. Check intent boundaries.'

@pytest.mark.parametrize('text',['<Experiences>','<Experiences>unstructured unfinished','<Experiences>1. x<Other>y','<Experiences>1. x<Experiences>2. y','</Experiences><Experiences>1. x'])
def test_advantage_refuses_ambiguous_or_empty_heading(text):
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import semantic_advantage
    with pytest.raises(LearningError):semantic_advantage(text)
