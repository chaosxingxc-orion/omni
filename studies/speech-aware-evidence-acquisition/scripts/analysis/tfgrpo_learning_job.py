"""Freeze one real H1 TF-GRPO learning job; preserve H2 for development.

Only the intersection of source-audited mixed groups and fixed A0-error roster
is admitted. Raw learning data stays under the data root, never in Git.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical,sha,validate_experience
from speech_aware_evidence_acquisition.rails.tfgrpo_prompts import SOURCE_SHA256,UPSTREAM_COMMIT
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
NAME='slurp-tfgrpo-fixed-p-h1-learning-v1'
RUN='SAEA-SLURP-TFGRPO-FIXED-P-H1-LEARNING-V1'
AUTH='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'
AGENT_OBJECTIVE='Infer the spoken intent and exact slots from the supplied Perception result and retrieved evidence.'
LEARNING_OBJECTIVE=('Improve evidence consumption and intent/slot decisions using contrasts between successful and failed Solver outputs. '
    'Extract conditional transferable procedures, not sample-specific answers, names, numbers or memorized labels. '
    'Audio is represented only by hashes; do not claim to hear it or infer unavailable acoustic facts.')


def read(path):return json.loads(Path(path).read_bytes())
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path,root,kind):return {'root':kind,'relpath':str(Path(path).resolve().relative_to(Path(root).resolve())),'sha256':file_sha(path)}
def raw(doc):return (json.dumps(doc,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()


def select(half,roster):
    if (half.get('schema')!='saea-tfgrpo-seed-half-v1' or half.get('source_half')!='h1'
            or half.get('evaluation_half')!='h2' or half.get('rollout_kind')!='shared-perception-adaptation'):
        raise ValueError('first job is fixed-P H1 learning with H2 held out')
    allowed={r['sample_id']:r for r in roster['items']}
    if len(allowed)!=len(roster['items']) or not allowed:raise ValueError('unique fixed error roster required')
    groups=[copy.deepcopy(g) for g in half['groups'] if g['query_id'] in allowed]
    ids=[g['query_id'] for g in groups]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('nonempty unique error intersection required')
    validate_experience(half['parent'],half['model_id'])
    if sha(half['policy'])!=half['policy_sha256']:raise ValueError('exported policy changed')
    custody={r['trajectory_id']:r for r in half['custody']}
    seen=set()
    for g in groups:
        if {r['reward'] for r in g['rollouts']}!={0,1} or len(g['rollouts'])!=4:
            raise ValueError('four genuinely mixed binary rollouts required')
        for r in g['rollouts']:
            rid=r['trajectory_id']
            if rid in seen or rid not in custody:raise ValueError('trajectory custody absent or duplicated')
            seen.add(rid)
            if (r['model_id'],r['policy_sha256'],r['experience_sha256'])!=(half['model_id'],half['policy_sha256'],half['parent']['sha256']):
                raise ValueError('rollout model/policy/experience differs')
    items=[copy.deepcopy(allowed[sid]) for sid in ids]
    construction={'schema':'saea-roster-receipt-v1','roster_id':RUN,'roster_role':'construction',
        'carrier_lock_key':'slurp','split_role':'discovery','split_identity_hash':split_identity_hash(prefixed('slurp',ids)),
        'count':len(ids),'items':items,'selection':'Exact H1 mixed candidate groups intersected with fixed A0-error439; no draw.',
        'evaluation_half':'h2','source_queries_must_not_be_reported_as_held_out':True}
    return groups,construction


def documents(half,roster,*,runtime_identity,roster_sha256,provenance,concurrency=128):
    from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import learning_configuration,validate_registration
    groups,construction=select(half,roster)
    contacts=[]
    for g in groups:
        contacts.extend({'stage':'summary','identity':r['trajectory_id'],'sample_id':g['query_id']} for r in g['rollouts'])
        contacts.extend({'stage':s,'identity':g['query_id'],'sample_id':g['query_id']} for s in ('advantage','group_update'))
    contacts.append({'stage':'batch_update','identity':'batch','sample_id':groups[0]['query_id']})
    count=len(contacts)
    # Positive speech cap is retained only for legacy ExecutionPlan schema; text
    # contacts carry no audio and actual audio usage must remain zero.
    plan=ExecutionPlan(run_id=RUN,attempt_id='learning-v1',execution_profile='bounded-discovery-probe',
        carrier_lock_key='slurp',split_role='discovery',split_identity_hash=construction['split_identity_hash'],
        planned_model_calls=count,planned_gpu_hours=8.0,planned_speech_audio_seconds=1,protocol_hash='0'*64)
    reg={'schema':'saea-tfgrpo-runtime-binding-v1','plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'},
        'model_id':half['model_id'],'construction_roster_sha256':roster_sha256,'contacts':contacts,
        'decoding_params':{'temperature':0,'seed':20261101,'max_tokens':4096},'max_context_bytes':2000000,
        'runtime_identity_sha256':sha(runtime_identity)}
    if runtime_identity.get('model_fragment')!=half['model_id']:raise ValueError('same frozen resident model required')
    reg['learning_config_sha256']=sha(learning_configuration(reg))
    common={'agent_objective':AGENT_OBJECTIVE,'learning_objective':LEARNING_OBJECTIVE,'num_experiences':2}
    bounds={'max_calls':count,'concurrency':concurrency,'max_operations':128,'max_items':128,'max_characters':16000}
    batch={'schema':'saea-tfgrpo-batch-v1','groups':groups,'parent':half['parent'],'common':common,
        'model_id':half['model_id'],'policy_sha256':half['policy_sha256'],'rollout_kind':half['rollout_kind'],
        'learning_config_sha256':reg['learning_config_sha256'],'prompt_sha256':SOURCE_SHA256,'upstream_commit':UPSTREAM_COMMIT,'bounds':bounds}
    reg['batch_sha256']=sha(batch)
    validate_registration(reg,plan=plan,model_id=half['model_id'])
    source={'schema':'saea-tfgrpo-learning-source-v1','batch':batch,'provenance':provenance}
    return construction,source,reg,plan


def prepare(repo,data,exports,hardset,out,runtime_identity):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned job directory required')
    manifest=read(exports/'complete.json')
    if manifest['schema']!='saea-tfgrpo-seed-groups-v1' or any(manifest[k]!=0 for k in ('new_model_calls','new_tokenizer_calls','new_scoring_jobs')):
        raise ValueError('source-audited model-free export required')
    for name,digest in manifest['artifacts'].items():
        if file_sha(exports/name)!=digest:raise ValueError('source export changed')
    half=read(exports/'h1.json');h2=read(exports/'h2.json');roster=read(hardset)
    if len(roster['items'])!=439 or len(half['groups'])!=52 or len(h2['groups'])!=48:
        raise ValueError('fixed audited625 export and439 error roster required')
    if set(g['query_id'] for g in half['groups']) & set(g['query_id'] for g in h2['groups']):raise ValueError('learning/development overlap')
    groups,construction=select(half,roster)
    roster_bytes=raw(construction);roster_sha=hashlib.sha256(roster_bytes).hexdigest()
    provenance=[pin(exports/'complete.json',data,'data'),pin(exports/'h1.json',data,'data'),pin(hardset,repo,'repository')]
    identity=read(runtime_identity)
    construction,source,reg,plan=documents(half,roster,runtime_identity=identity,roster_sha256=roster_sha,provenance=provenance)
    out.mkdir(parents=True)
    for name,doc in [('construction-roster.json',construction),('source.json',source),('registration.json',reg),('runtime-identity.json',identity)]:publish_new(out/name,raw(doc))
    result={'schema':'saea-tfgrpo-learning-job-v1','job':NAME,'run_id':RUN,'source_half':'h1','evaluation_half':'h2',
        'groups':len(groups),'trajectories':4*len(groups),'generation_calls':plan.planned_model_calls,'fit_calls':plan.planned_model_calls,
        'excluded_mixed_but_not_a0_error':len(half['groups'])-len(groups),'rollout_kind':'shared-perception-adaptation',
        'new_model_calls':0,'new_tokenizer_calls':0,'batch_sha256':sha(source['batch']),
        'runtime_identity_sha256':sha(identity),'artifacts':{n:file_sha(out/n) for n in ('construction-roster.json','source.json','registration.json','runtime-identity.json')},
        'inputs':provenance,'development_queries_used_for_learning':False,'readiness':'prepared-not-admitted'}
    publish_new(out/'complete.json',raw(result));return result


def deployment_identity(repo,model_id,identity):
    """Derive immutable serving identity without FlightRuntime._prepare bootstrap."""
    from speech_aware_evidence_acquisition.core.config import read_fragment,resolve_runtime_receipt_path
    from speech_aware_evidence_acquisition.core.session.vllm_resident import required_identity
    fragment=read_fragment(repo,'model',model_id)
    receipt=resolve_runtime_receipt_path(repo,fragment)
    return required_identity(fragment=fragment,model_fragment=model_id,runtime_receipt=read(receipt),
        receipt_path=receipt,host=identity['host'],port=identity['port'])


def verify_job(repo,data,job):
    """Verify derivation, not only internally self-consistent hashes, before writes."""
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import read_pin,load_dynamic_source
    complete=read(job/'complete.json')
    required={'construction-roster.json','source.json','registration.json','runtime-identity.json'}
    if set(complete['artifacts'])!=required:raise ValueError('exact prepared artifact set required')
    for name,digest in complete['artifacts'].items():
        if file_sha(job/name)!=digest:raise ValueError('prepared learning job changed')
    reg=read(job/'registration.json');roster=read(job/'construction-roster.json')
    source=read(job/'source.json');identity=read(job/'runtime-identity.json')
    if (sha(identity)!=reg['runtime_identity_sha256'] or sha(identity)!=complete['runtime_identity_sha256']
            or deployment_identity(repo,reg['model_id'],identity)!=identity):
        raise ValueError('prepared deployment identity differs')
    if source['provenance']!=complete['inputs'] or len(source['provenance'])!=3:
        raise ValueError('exact export provenance required')
    manifest,half,hardset=[read_pin(p,repo_root=repo,data_root=data) for p in source['provenance']]
    if (manifest['schema']!='saea-tfgrpo-seed-groups-v1'
            or manifest['artifacts']['h1.json']!=source['provenance'][1]['sha256']):
        raise ValueError('export manifest differs')
    expected_roster,expected_source,expected_reg,plan=documents(half,hardset,runtime_identity=identity,
        roster_sha256=file_sha(job/'construction-roster.json'),provenance=source['provenance'],
        concurrency=source['batch']['bounds']['concurrency'])
    if (roster,source,reg)!=(expected_roster,expected_source,expected_reg):
        raise ValueError('prepared job differs from exact source derivation')
    if (complete['batch_sha256']!=reg['batch_sha256'] or complete['groups']!=len(roster['items'])
            or complete['generation_calls']!=len(reg['contacts']) or complete['fit_calls']!=len(reg['contacts'])):
        raise ValueError('prepared job counts differ')
    load_dynamic_source({'prepared_training_feedback':{'schema':'saea-prepared-tfgrpo-dynamic-v1',
        'source':pin(job/'source.json',data,'data')},'tfgrpo_learning_registration':reg},plan=plan,
        repo_root=repo,data_root=data,sample_ids=tuple(x['sample_id'] for x in roster['items']),
        roster_sha256=file_sha(job/'construction-roster.json'))
    return complete,reg,roster


def register(repo,data,job):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment();require_clean_study_repo(repo)
    complete,reg,roster=verify_job(repo,data,job)
    names=[RUN,RUN+'-FIT'];ledger=repo/'docs/exposure-ledger.md'
    if any('| '+rid+' |' in ledger.read_text() or (data/'attempts'/rid).exists() for rid in names):raise ValueError('existing learning registration or spend')
    model_id=reg['model_id'];model=read(repo/f'configs/model/{model_id}.json')
    if reg['batch_sha256']!=complete['batch_sha256'] or len(reg['contacts'])!=complete['generation_calls']:
        raise ValueError('job registration and source budget differ')
    roster_rel=f'docs/receipts/rosters/{NAME}.json'
    baseline=read(repo/'configs/baseline/system-group-update-text-v1.json');baseline.pop('whole_roster_wave_barrier',None)
    baseline['what_this_is']='Model-driven TF-GRPO fixed-P Solver experience learning; H1 A0-error mixed groups only.'
    baseline.pop('phase_material',None)
    from speech_aware_evidence_acquisition.rails.prepared_text import PREPARED_TEXT_INSTRUCTION
    baseline['task_instruction']=PREPARED_TEXT_INSTRUCTION
    baseline['serving_slot_tokens']=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
    baseline['serving_slots']=read(repo/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json')['serving_slots']
    exp=read(repo/'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json')
    for k in ('population','stage3_measurement','prepared_phase_supply','native_episode','knowledge_stores','live_phase_fit','native_contact_fit'):
        exp.pop(k,None)
    exp.update(experiment_id=RUN,purpose=baseline['what_this_is'],execution_plan={k:reg['plan'][k] for k in ('execution_profile','split_role','planned_model_calls','planned_gpu_hours','planned_speech_audio_seconds')},
        tfgrpo_learning_registration=reg,prepared_training_feedback={'schema':'saea-prepared-tfgrpo-dynamic-v1','source':pin(job/'source.json',data,'data')},
        artifact_result={'schema':'saea-artifact-result-binding-v1','role':'training-feedback-proposal','construction_roster':{'repo_relpath':roster_rel,'sha256':file_sha(job/'construction-roster.json')}},
        tfgrpo_contact_fit={'run_id':RUN+'-FIT','attempt_id':'learning-fit-v1','contact_cap':complete['fit_calls']},
        stage3_algorithm={'id':'tfgrpo-system-memory-disjoint-v1','operators_used':['credit_assignment'],'note':'Official four-stage model-driven update, shared-P adaptation; candidate only.'})
    cell,=exp['cells'];cell.update(cell_id=NAME,run_id=RUN,attempt_id='learning-v1',baseline_fragment=NAME,
        allowed_execution_stages=['check','worker','campaign'],readiness={'state':'held','requirements':[],'reason':'Registered exact learning source; await shared live runtime preflight and lease.'})
    exp['populations'][0]['ids']=[x['sample_id'] for x in roster['items']]
    from tfrl_current_generation import SESSION
    exp['service_session']=SESSION
    for rel,doc in [(roster_rel,roster),(f'configs/baseline/{NAME}.json',baseline),(f'configs/experiment/{NAME}.json',exp)]:
        path=repo/rel
        if path.exists():raise ValueError('learning configuration already exists')
        path.parent.mkdir(parents=True,exist_ok=True);publish_new(path,raw(doc))
    cfg=compose(repo,model_id,cell['dataset_fragment'],NAME,NAME);ph=protocol_hash(cfg.config_hash,cfg.values['arm'],roster['split_identity_hash'])
    package=repo/'docs/readiness'/NAME;package.mkdir(parents=True)
    rows=[]
    for kind,run,attempt in [('generation',RUN,'learning-v1'),('fit',RUN+'-FIT','learning-fit-v1')]:
        plan=ExecutionPlan(**{**reg['plan'],'run_id':run,'attempt_id':attempt,'protocol_hash':ph}).validate(source_root=repo)
        doc={'schema':'saea-tfgrpo-registered-plan-v1','execution_plan':asdict(plan),'job_manifest':pin(job/'complete.json',data,'data'),
             'config_sha256':cfg.config_hash,'authority':AUTH,'actual_audio_seconds_expected':0,'readiness':'held'}
        path=package/(kind+'-plan.json');publish_new(path,raw(doc))
        rows.append(['2026-11-01',run,'bounded-discovery-probe','TF-GRPO H1 fixed-P Solver learning '+kind,model_id,'slurp TRAIN H1 error mixed groups','discovery',roster['split_identity_hash'],'no',str(plan.planned_model_calls),'1','8.0','All source rollout costs retained',str(path.relative_to(repo)),f'Plan SHA256 {file_sha(path)}; root countersigned under {AUTH}; no H2 feedback in learning; held; text only, actual audio zero'])
    ledger.write_text(ledger.read_text()+'\n## Fixed-P TF-GRPO learning\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n')
    return {'job':NAME,'rows':complete['groups'],'calls':complete['generation_calls'],'fit_calls':complete['fit_calls'],'readiness':'held','new_model_calls':0}


def admit(repo,data,job):
    """CPU-only admission against the existing resident; never acquires a card."""
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.session.serving import find_resident_session,assert_identity_matches
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash,resolve_runtime_receipt_path,read_fragment
    from speech_aware_evidence_acquisition.core.flight.repin import apply_decision_and_readiness
    sys.path.insert(0,str(repo/'scripts/flights/direct'))
    from thinking_system_mint_receipt import mint
    from tfrl_current_generation import SESSION
    assert_execution_environment();require_clean_study_repo(repo)
    complete,reg,roster=verify_job(repo,data,job)
    pkg=repo/'docs/readiness'/NAME
    exp=read(repo/f'configs/experiment/{NAME}.json');cell,=exp['cells']
    if cell['readiness']['state']!='held' or exp['tfgrpo_learning_registration']!=reg:
        raise ValueError('exact held registration required')
    resident=find_resident_session(data)
    if resident is None or resident[1]['session_id']!=SESSION:raise ValueError('expected resident absent')
    assert_identity_matches(resident[1],read(job/'runtime-identity.json'))
    receipt=resolve_runtime_receipt_path(repo,read_fragment(repo,'model',reg['model_id']))
    receipt_doc=read(receipt)
    for instance in receipt_doc['deployment']['instances']:
        session=SessionReceipt.from_json(read(data/'serving'/instance['instance_id']/'session-receipt.json'))
        verify_vllm_session(session,receipt_doc)
    runs=[RUN,RUN+'-FIT'];decision_id=RUN+'-ADMISSION';decision_rel='coordination/decisions/'+decision_id+'.json'
    if any((data/'attempts'/rid).exists() for rid in runs):raise ValueError('existing learning spend')
    draft={'schema':'saea-campaign-decision-v1','status':'DRAFT-NOT-MINTED','decision_id':decision_id,
        'eligible_run_ids':runs,'eligible_attempt_ids':['learning-v1','learning-fit-v1'],'authorization_basis':AUTH,
        'owner':'root','decision':'Execute the registered 32 H1 mixed-error groups through shared TF-GRPO runtime; candidate only, no H2 feedback.',
        'antecedents':{'owner':{'path':AUTH,'sha256':file_sha(repo/AUTH)},
            'runtime':{'path':str(receipt.relative_to(repo)),'sha256':file_sha(receipt)}},
        'source_manifest':pin(job/'complete.json',data,'data'),'service':{'expected_session_id':SESSION},
        'budget':{k:read(pkg/(k+'-plan.json'))['execution_plan'] for k in ('generation','fit')}}
    publish_new(pkg/'root-admission-draft-v1.json',raw(draft))
    family=SimpleNamespace(PACKAGE=str(pkg.relative_to(repo)),DECISION_ID=decision_id,DECISION_RELPATH=decision_rel,eligible_run_ids=lambda:runs)
    result=mint(str(repo),str(data),'Codex',False,family=family,tool='scripts/analysis/tfgrpo_learning_job.py')
    apply_decision_and_readiness(repo_root=repo,data_root=data,experiment=NAME,cell_id=NAME,
        decision=(decision_rel,decision_id),set_ready=True)
    cfg=compose(repo,reg['model_id'],cell['dataset_fragment'],NAME,NAME)
    ph=protocol_hash(cfg.config_hash,cfg.values['arm'],roster['split_identity_hash'])
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    for kind in ('generation','fit'):
        path=pkg/(kind+'-plan.json');old=file_sha(path);doc=read(path)
        doc['execution_plan']['protocol_hash']=ph;doc['config_sha256']=cfg.config_hash;doc['readiness']='admitted-pending-clean-commit'
        path.write_bytes(raw(doc));text=text.replace(old,file_sha(path))
    ledger.write_text(text)
    return {**result,'new_model_calls':0,'remaining':['commit admission','shared preflight','generation lease after predecessor release']}


def verify_release(path,data):
    """A release event proves sealed generation, never merely an exited process."""
    doc=read(path)
    if (doc.get('schema')!='saea-generation-released-v1' or doc.get('generation_sealed') is not True
            or doc.get('generation_lease_released') is not True
            or doc.get('run_id')!='SAEA-SLURP-TFRL-HARDSET-A0-OBSERVATION-R1-V1'):
        raise ValueError('exact A0 generation release event required')
    for pin_doc in doc['artifacts'].values():
        p=Path(pin_doc['path']).resolve()
        if not p.is_relative_to(data.resolve()) or file_sha(p)!=pin_doc['sha256']:
            raise ValueError('predecessor sealed artifact differs')
    if not doc['artifacts']:raise ValueError('empty predecessor release')
    return doc


def run(repo,data,job,out,after):
    """One detached dispatch; campaign alone acquires and releases the lease."""
    import time
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells,declared_readiness,run_campaign
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from run_real_engine_candidate import assert_campaign_success
    from tfrl_current_generation import SESSION
    assert_execution_environment();commit=require_clean_study_repo(repo)
    complete,reg,roster=verify_job(repo,data,job)
    verify_release(after,data)
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned output required; no blind retry')
    cells=load_campaign_cells(repo,NAME);cell,=cells
    if declared_readiness(cell,data).state!='ready':raise ValueError('learning job remains held')
    for rid,aid in ((RUN,'learning-v1'),(RUN+'-FIT','learning-fit-v1')):
        if (data/'attempts'/rid/(aid+'.json')).exists():raise ValueError('existing attempt; reconcile before resume')
    out.mkdir(parents=True);publish_new(out/'started.json',raw({'study_commit':commit,'job':NAME,'predecessor_release_sha256':file_sha(after)}))
    try:
        runtime=build_flight_runtime(repo,data,model=reg['model_id'],dataset=cell.document['dataset_fragment'],
            baseline=NAME,experiment=NAME,run_id=RUN,attempt_id='learning-v1',runtime_root=data/'work/scratch/runtime-root',
            lock_path=repo.parents[1]/'docs/datasets.lock.json')
        result=run_campaign([runtime],cells=cells,repo_root=repo,data_root=data,owner='Codex-TFGRPO',expected_session_id=SESSION)
        publish_new(out/'campaign.json',raw(result));assert_campaign_success(result,RUN)
        publish_new(out/'complete.json',raw({'job':NAME,'groups':complete['groups'],'result':result,'candidate_accepted':False}))
        return {'completed':True,'candidate_accepted':False}
    except BaseException as error:
        publish_new(out/'failed.json',raw({'type':type(error).__name__,'error':str(error),'automatic_retry':False,
            'preserve':'learning receipts, pending contact intents, FIT and generation costs'}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','register','verify','admit','run'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('exports','hardset','out','runtime-identity','job','after'):p.add_argument('--'+k,type=Path)
    a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='prepare':result=prepare(**{k:a[k] for k in ('repo','data','exports','hardset','out','runtime_identity')})
    elif mode=='verify':
        result=verify_job(a['repo'],a['data'],a['job'])[0]
    elif mode=='run':result=run(**{k:a[k] for k in ('repo','data','job','out','after')})
    else:result=globals()[mode](a['repo'],a['data'],a['job'])
    print(json.dumps(result,sort_keys=True))
