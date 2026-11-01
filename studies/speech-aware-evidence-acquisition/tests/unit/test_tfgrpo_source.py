"""Dynamic source identity and finite stage contracts reject drift before calls."""
import copy,json,hashlib
from dataclasses import asdict,replace
from types import SimpleNamespace
import pytest
from test_rails_frozen_core_client import _plan
from test_tfgrpo_runtime import config_hash,runtime_hash
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience,sha,LearningError
from speech_aware_evidence_acquisition.rails.tfgrpo_prompts import SOURCE_SHA256,UPSTREAM_COMMIT
from speech_aware_evidence_acquisition.rails.tfgrpo_source import load_dynamic_source


def setup(tmp_path):
    sid='slurp:train:001';model='frozen-model';parent=empty_experience(model)
    rolls=[{'trajectory_id':r,'trajectory':'Verified trace.','critique':'Reference checked.',
        'reward':v,'model_id':model,'policy_sha256':'f'*64,'experience_sha256':parent['sha256']}
        for r,v in [('r1',0),('r2',1)]]
    batch={'schema':'saea-tfgrpo-batch-v1','groups':[{'query_id':sid,'query':'Task','answer':'Reference','rollouts':rolls}],
        'parent':parent,'common':{'agent_objective':'Task','learning_objective':'Evidence','num_experiences':2},
        'model_id':model,'policy_sha256':'f'*64,'rollout_kind':'shared-perception-adaptation',
        'learning_config_sha256':config_hash(),'prompt_sha256':SOURCE_SHA256,'upstream_commit':UPSTREAM_COMMIT,
        'bounds':{'max_calls':5,'concurrency':2,'max_operations':32,'max_items':128,'max_characters':32000}}
    contacts=[{'stage':s,'identity':i,'sample_id':sid} for s,i in [('summary','r1'),('summary','r2'),
        ('advantage',sid),('group_update',sid),('batch_update','batch')]]
    plan=replace(_plan(),carrier_lock_key='slurp')
    reg={'schema':'saea-tfgrpo-runtime-binding-v1','plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'},
         'model_id':model,'batch_sha256':sha(batch),'construction_roster_sha256':'c'*64,'contacts':contacts,
         'decoding_params':{'max_tokens':100,'temperature':0},'max_context_bytes':10000,'learning_config_sha256':config_hash(),'runtime_identity_sha256':runtime_hash()}
    (tmp_path/'evidence.json').write_text('{"verified":true}')
    source={'schema':'saea-tfgrpo-learning-source-v1','batch':batch,'provenance':[{'root':'data','relpath':'evidence.json',
        'sha256':hashlib.sha256((tmp_path/'evidence.json').read_bytes()).hexdigest()}]}
    (tmp_path/'source.json').write_text(json.dumps(source))
    values={'tfgrpo_learning_registration':reg,'prepared_training_feedback':{'schema':'saea-prepared-tfgrpo-dynamic-v1',
        'source':{'root':'data','relpath':'source.json','sha256':hashlib.sha256((tmp_path/'source.json').read_bytes()).hexdigest()}}}
    args={'plan':plan,'repo_root':tmp_path,'data_root':tmp_path,'sample_ids':(sid,),'roster_sha256':'c'*64}
    return values,args,source


def test_dynamic_source_exact_batch_and_finite_contacts(tmp_path):
    values,args,expected=setup(tmp_path)
    source,receipt=load_dynamic_source(values,**args)
    assert source==expected and receipt['contacts']==5 and receipt['rows']==1


@pytest.mark.parametrize('change', ['source','provenance','roster','contact','configuration'])
def test_dynamic_source_drift_refused(tmp_path,change):
    values,args,_=setup(tmp_path)
    if change=='source':(tmp_path/'source.json').write_text('{}')
    elif change=='provenance':(tmp_path/'evidence.json').write_text('{}')
    elif change=='roster':args['sample_ids']=('slurp:train:002',)
    elif change=='contact':values['tfgrpo_learning_registration']['contacts'][0]['identity']='unregistered'
    else:values['tfgrpo_learning_registration']['decoding_params']['max_tokens']=200
    with pytest.raises(LearningError):load_dynamic_source(values,**args)


def test_real_artifact_binding_accepts_only_dynamic_fit_contract(tmp_path):
    from speech_aware_evidence_acquisition.core.flight.runtime import validate_artifact_result_binding, FlightRuntimeError
    from speech_aware_evidence_acquisition.rails.prepared_text import PREPARED_TEXT_INSTRUCTION
    values,args,source=setup(tmp_path)
    from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
    sid=args['sample_ids'][0];plan=replace(args['plan'],split_identity_hash=split_identity_hash(prefixed('slurp',(args['sample_ids'][0],))))
    values['tfgrpo_learning_registration']['plan']={k:v for k,v in asdict(plan).items() if k!='protocol_hash'}
    roster={'schema':'saea-roster-receipt-v1','roster_id':'TRAIN-TFGRPO','carrier_lock_key':'slurp',
        'split_role':'discovery','roster_role':'construction','seed':1,'per_stratum':{'all':1},
        'strata_counts':{'all':1},'items':[{'sample_id':sid,'stratum':'all'}],
        'excluded_count':0,'split_identity_hash':plan.split_identity_hash}
    (tmp_path/'roster.json').write_text(json.dumps(roster));rh=hashlib.sha256((tmp_path/'roster.json').read_bytes()).hexdigest()
    values['tfgrpo_learning_registration']['construction_roster_sha256']=rh
    values.update(arm='prepared-context-text',policy='phase-direct-only-v1',transport='vllm',result_kind='artifact',
        task_instruction=PREPARED_TEXT_INSTRUCTION,evidence_source='none',scorers=[],
        tfgrpo_contact_fit={'run_id':'fit-run','attempt_id':'fit-attempt','contact_cap':5},
        artifact_result={'schema':'saea-artifact-result-binding-v1','role':'training-feedback-proposal',
            'construction_roster':{'repo_relpath':'roster.json','sha256':rh}})
    authority=validate_artifact_result_binding(values,plan=plan,repo_root=tmp_path,data_root=tmp_path,sample_ids=(sid,))
    assert authority['construction_roster']['sha256']==rh
    values['tfgrpo_contact_fit']['contact_cap']=4
    with pytest.raises(FlightRuntimeError,match='one endpoint'):
        validate_artifact_result_binding(values,plan=plan,repo_root=tmp_path,data_root=tmp_path,sample_ids=(sid,))


def test_registered_entry_executes_pinned_batch_and_closes_client(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.rails import tfgrpo_runtime
    values,args,source=setup(tmp_path);calls=[];closed=[]
    class Caller:
        client=SimpleNamespace(close=lambda:closed.append(True))
        def preflight(self,request):assert request['learning_config_sha256']==config_hash()
        def reconcile(self,request):raise AssertionError('no prior contact')
        def __call__(self,request):
            calls.append(request)
            text=('Summary' if request['stage']=='summary' else '<Experiences>Check evidence.</Experiences>'
                  if request['stage']=='advantage' else '[{"operation":"ADD","id":null,"content":"Check evidence."}]')
            return {'text':text,'model_id':'frozen-model','receipt_id':str(len(calls)),
                    'usage':{'calls':1,'prompt_tokens':10,'completion_tokens':3}}
    runtime=SimpleNamespace(_composed=SimpleNamespace(values=values),plan=args['plan'],_repo_root=tmp_path,
        _data_root=tmp_path,_samples=[SimpleNamespace(sample_id=args['sample_ids'][0])])
    monkeypatch.setattr(tfgrpo_runtime,'bind_runtime',lambda *a,**k:Caller())
    candidate=tfgrpo_runtime.run_registered_learning(runtime,receipt_root=tmp_path/'runtime',learner_root=tmp_path/'learner')
    assert len(calls)==5 and closed==[True]
    assert candidate['learning_usage']['calls']==5 and candidate['deployment_accepted'] is False


def final_setup(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.rails import tfgrpo_final_import
    values,args,source=setup(tmp_path)
    batch=source['batch'];other=copy.deepcopy(batch['groups'][0]);other['query_id']='slurp:train:002'
    for row in other['rollouts']:row['trajectory_id']+='-other'
    batch['groups'].append(other)
    batch['bounds'].update(max_calls=1,concurrency=1)
    batch['final_import']={'relpath':'historical-import.json','sha256':'e'*64}
    reg=values['tfgrpo_learning_registration'];reg['contacts']=reg['contacts'][-1:];reg['batch_sha256']=sha(batch)
    path=tmp_path/'source.json';path.write_text(json.dumps(source))
    values['prepared_training_feedback']['source']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    seen=[]
    def verified(spec,**kwargs):
        seen.append(kwargs)
        return [],{'source':spec,'learned_query_ids':sorted(g['query_id'] for g in kwargs['groups']),
                   'imported_group_updates':2,'operations_sha256':sha([])}
    monkeypatch.setattr(tfgrpo_final_import,'load_import',verified)
    return values,args,source,seen


def test_final_source_one_runtime_anchor_preserves_full_learning_population(tmp_path,monkeypatch):
    values,args,source,seen=final_setup(tmp_path,monkeypatch)
    actual,receipt=load_dynamic_source(values,**args)
    assert actual==source and receipt['contacts']==1 and receipt['rows']==1
    assert len(seen)==1 and len(seen[0]['groups'])==2


def test_final_source_import_refusal_precedes_generation(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.rails import tfgrpo_final_import
    values,args,_,_=final_setup(tmp_path,monkeypatch)
    def refuse(*a,**k):raise LearningError('historical custody invalid')
    monkeypatch.setattr(tfgrpo_final_import,'load_import',refuse)
    with pytest.raises(LearningError,match='historical custody'):load_dynamic_source(values,**args)


def test_final_runtime_only_calls_merge_with_all_learning_ids(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.rails import tfgrpo_runtime
    values,args,_,seen=final_setup(tmp_path,monkeypatch);calls=[]
    class Caller:
        client=SimpleNamespace(close=lambda:None)
        def preflight(self,request):pass
        def reconcile(self,request):raise AssertionError('no prior contact')
        def __call__(self,request):
            calls.append(request)
            return {'text':'[]','model_id':'frozen-model','receipt_id':'actual-final-1',
                    'usage':{'calls':1,'prompt_tokens':10,'completion_tokens':3}}
    runtime=SimpleNamespace(_composed=SimpleNamespace(values=values),plan=args['plan'],_repo_root=tmp_path,
        _data_root=tmp_path,_samples=[SimpleNamespace(sample_id=args['sample_ids'][0])])
    monkeypatch.setattr(tfgrpo_runtime,'bind_runtime',lambda *a,**k:Caller())
    candidate=tfgrpo_runtime.run_registered_learning(runtime,receipt_root=tmp_path/'runtime',learner_root=tmp_path/'learner')
    assert [r['stage'] for r in calls]==['batch_update']
    assert candidate['learned_query_ids']==['slurp:train:001','slurp:train:002']
    assert candidate['learning_usage']['calls']==1 and len(seen)==2
