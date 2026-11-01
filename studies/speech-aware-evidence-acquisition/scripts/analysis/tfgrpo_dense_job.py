"""Prepare source-derived dense TF-GRPO registration; no inference authorization."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import tfgrpo_learning_job as base
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    DENSE_REWARD,canonical,sha,validate_experience,intent_slot_reward,group_has_contrast)
from speech_aware_evidence_acquisition.rails.tfgrpo_prompts import SOURCE_SHA256,UPSTREAM_COMMIT
from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import learning_configuration,validate_registration
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
NAME='slurp-tfgrpo-dense-h1-allwrong-v1'
RUN='SAEA-SLURP-TFGRPO-DENSE-H1-ALLWRONG-V1'
OBJECTIVE=('Learn conditional intent and slot procedures from relative partial correctness. '
    'All source rollouts fail strict task success; a better partial reward is not a successful answer. '
    'Reward describes the answer after fixed delivery normalization; do not attribute its changes to raw reasoning. '
    'Do not memorize sample-specific entities, names, numbers, or answers. '
    'Audio is unavailable to this text reflector; do not invent acoustic evidence.')


def documents(half,identity,model,provenance,*,run_id=RUN,concurrency=128):
    if (half.get('schema')!='saea-tfgrpo-dense-h1-groups-v1' or half.get('reward_mode')!=DENSE_REWARD
            or half.get('source_half')!='h1' or half.get('evaluation_half')!='h2'
            or half.get('rollout_kind')!='shared-perception-adaptation'):
        raise ValueError('exact H1 dense source required')
    validate_experience(half['parent'],half['model_id'])
    if half['parent']['items'] or half['parent']['learned_query_ids'] or sha(half['policy'])!=half['policy_sha256']:
        raise ValueError('historical empty-experience source policy required')
    groups=copy.deepcopy(half['groups']);ids=[g['query_id'] for g in groups]
    if len(ids)!=51 or len(set(ids))!=51:raise ValueError('exact fixed51 groups required')
    custody={c['trajectory_id']:c for c in half['custody']}
    if len(custody)!=204 or len(half['custody'])!=204:raise ValueError('204 unique source receipts required')
    seen=set()
    for g in groups:
        rolls=g['rollouts']
        if len(rolls)!=4 or not group_has_contrast(rolls,DENSE_REWARD) or any(r['reward']==1 for r in rolls):
            raise ValueError('allwrong partial contrast required')
        for r in rolls:
            rid=r['trajectory_id'];crit=json.loads(r['critique'])
            if (rid in seen or rid not in custody or r['reward']!=intent_slot_reward(crit)
                    or crit.get('strict_task_success')!=0 or crit.get('reward_mode')!=DENSE_REWARD
                    or (r['model_id'],r['policy_sha256'],r['experience_sha256'])!=(half['model_id'],half['policy_sha256'],half['parent']['sha256'])):
                raise ValueError('source reward or rollout provenance differs')
            trajectory=json.loads(r['trajectory'])
            normalized=trajectory.get('scored_answer_after_fixed_normalization')
            if (not isinstance(normalized,str) or sha_bytes(normalized.encode())!=crit.get('scored_answer_sha256')
                    or crit['scored_answer_sha256']!=custody[rid].get('scored_answer_sha256')):
                raise ValueError('exact scored answer and normalization custody required')
            seen.add(rid)
    items=copy.deepcopy(half['roster_items'])
    if [r['sample_id'] for r in items]!=ids:raise ValueError('source roster order differs')
    roster={'schema':'saea-roster-receipt-v1','roster_id':run_id,'roster_role':'construction','carrier_lock_key':'slurp',
        'split_role':'discovery','split_identity_hash':split_identity_hash(prefixed('slurp',ids)),
        'count':51,'items':items,'selection':'Fixed H1 historical allwrong groups with partial reward contrast; no draw.',
        'evaluation_half':'h2','source_queries_must_not_be_reported_as_held_out':True}
    common={'agent_objective':base.AGENT_OBJECTIVE,'learning_objective':OBJECTIVE,'num_experiences':2,'reward_mode':DENSE_REWARD}
    contacts=[]
    for g in groups:
        contacts.extend({'stage':'summary','identity':r['trajectory_id'],'sample_id':g['query_id']} for r in g['rollouts'])
        contacts.extend({'stage':stage,'identity':g['query_id'],'sample_id':g['query_id']} for stage in ('advantage','group_update'))
    contacts.append({'stage':'batch_update','identity':'batch','sample_id':ids[0]})
    plan=ExecutionPlan(run_id=run_id,attempt_id='learning-v1',execution_profile='bounded-discovery-probe',carrier_lock_key='slurp',
        split_role='discovery',split_identity_hash=roster['split_identity_hash'],planned_model_calls=307,planned_gpu_hours=8.0,
        planned_speech_audio_seconds=1,protocol_hash='0'*64)
    if identity.get('model_fragment')!=half['model_id']:raise ValueError('resident model name differs')
    reg={'schema':'saea-tfgrpo-runtime-binding-v1','plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'},
        'model_id':half['model_id'],'construction_roster_sha256':sha_bytes(base.raw(roster)), 'contacts':contacts,
        'decoding_params':base.learning_decoding(model),'max_context_bytes':2000000,'runtime_identity_sha256':sha(identity)}
    reg['learning_config_sha256']=sha(learning_configuration(reg))
    batch={'schema':'saea-tfgrpo-batch-v1','groups':groups,'parent':half['parent'],'common':common,'model_id':half['model_id'],
        'policy_sha256':half['policy_sha256'],'rollout_kind':half['rollout_kind'],'learning_config_sha256':reg['learning_config_sha256'],
        'prompt_sha256':SOURCE_SHA256,'upstream_commit':UPSTREAM_COMMIT,
        'bounds':{'max_calls':307,'concurrency':concurrency,'max_operations':128,'max_items':128,'max_characters':16000}}
    reg['batch_sha256']=sha(batch);validate_registration(reg,plan=plan,model_id=half['model_id'])
    return roster,{'schema':'saea-tfgrpo-learning-source-v1','batch':batch,'provenance':provenance},reg,plan


def sha_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def require_generation_budget(reg,model):
    """Fail before registration when full-batch consolidation is under-budgeted."""
    decoding=reg['decoding_params']
    if (decoding.get('max_tokens')!=16384 or decoding.get('thinking_budget_tokens')!=3072
            or model.get('answer_reserve_tokens')!=13312
            or model.get('reasoning_budget_tokens')!=3072):
        raise ValueError('dense generation requires explicit 16384 completion / 3072 reasoning / 13312 answer reserve')


def prepare(repo,data,exports,out,identity,service_session):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned prepared directory required')
    summary=base.read(exports/'complete.json');half=base.read(exports/'h1.json')
    if (summary['schema']!='saea-tfgrpo-dense-export-v1' or summary['selected_groups']!=51
            or summary['artifact_sha256']!=base.file_sha(exports/'h1.json')):raise ValueError('verified51 export required')
    identity=base.read(identity);model=base.read(repo/f"configs/model/{half['model_id']}.json")
    provenance=[base.pin(exports/'complete.json',data,'data'),base.pin(exports/'h1.json',data,'data')]
    roster,source,reg,plan=documents(half,identity,model,provenance)
    out.mkdir(parents=True)
    names={'construction-roster.json':roster,'source.json':source,'registration.json':reg,'runtime-identity.json':identity}
    for name,doc in names.items():publish_new(out/name,base.raw(doc))
    complete={'schema':'saea-tfgrpo-dense-job-v1','job':NAME,'run_id':RUN,'service_session':service_session,
        'groups':51,'trajectories':204,'generation_calls':307,'fit_calls':307,'new_model_calls':0,
        'reward_mode':DENSE_REWARD,'batch_sha256':reg['batch_sha256'],'runtime_identity_sha256':sha(identity),
        'artifacts':{n:base.file_sha(out/n) for n in names},'inputs':provenance,'readiness':'prepared-not-admitted'}
    publish_new(out/'complete.json',base.raw(complete));return complete


def verify_job(repo,data,job):
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import read_pin,load_dynamic_source
    complete=base.read(job/'complete.json')
    if complete.get('schema')!='saea-tfgrpo-dense-job-v1' or complete['job']!=NAME or complete['run_id']!=RUN:
        raise ValueError('exact dense job identity required')
    if set(complete['artifacts'])!={'construction-roster.json','source.json','registration.json','runtime-identity.json'}:
        raise ValueError('exact prepared artifact set required')
    for name,digest in complete['artifacts'].items():
        if base.file_sha(job/name)!=digest:raise ValueError('prepared job changed')
    roster,source,reg,identity=[base.read(job/n) for n in ('construction-roster.json','source.json','registration.json','runtime-identity.json')]
    if source['provenance']!=complete['inputs'] or len(source['provenance'])!=2:raise ValueError('exact export pins required')
    summary,half=[read_pin(p,repo_root=repo,data_root=data) for p in complete['inputs']]
    if summary['artifact_sha256']!=complete['inputs'][1]['sha256']:raise ValueError('export bytes differ')
    expected=documents(half,identity,base.read(repo/f"configs/model/{reg['model_id']}.json"),source['provenance'],concurrency=source['batch']['bounds']['concurrency'])
    if (roster,source,reg)!=expected[:3]:raise ValueError('prepared dense derivation differs')
    if base.deployment_identity(repo,reg['model_id'],identity)!=identity:raise ValueError('runtime identity differs')
    if any(complete[k]!=v for k,v in {'groups':51,'trajectories':204,'generation_calls':307,'fit_calls':307,
            'batch_sha256':reg['batch_sha256'],'runtime_identity_sha256':sha(identity),'reward_mode':DENSE_REWARD}.items()):
        raise ValueError('prepared counts or identity differ')
    load_dynamic_source({'prepared_training_feedback':{'schema':'saea-prepared-tfgrpo-dynamic-v1','source':base.pin(job/'source.json',data,'data')},
        'tfgrpo_learning_registration':reg},plan=expected[3],repo_root=repo,data_root=data,
        sample_ids=tuple(r['sample_id'] for r in roster['items']),roster_sha256=base.file_sha(job/'construction-roster.json'))
    return complete,reg,roster


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('exports','out','identity','job'):p.add_argument('--'+k,type=Path)
    p.add_argument('--service-session');a=p.parse_args()
    if a.mode=='prepare':r=prepare(a.repo,a.data,a.exports,a.out,a.identity,a.service_session)
    else:r=verify_job(a.repo,a.data,a.job)[0]
    print(json.dumps(r,sort_keys=True))
