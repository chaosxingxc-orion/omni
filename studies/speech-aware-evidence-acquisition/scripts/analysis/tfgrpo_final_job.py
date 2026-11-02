"""Register one metered final consolidation using verified completed TF-GRPO stages.

The new flight has one TRAIN anchor and one generation/FIT contact. All original
learning groups remain in its batch; their prior costs belong to import custody.
This tool never repairs truncated output or replays completed model requests.
"""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
import tfgrpo_learning_job as base
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha, validate_experience, BINARY_REWARD, group_has_contrast
from speech_aware_evidence_acquisition.rails.tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT

NAME = 'slurp-tfgrpo-fixed-p-h1-final-only-v1'
RUN = 'SAEA-SLURP-TFGRPO-FIXED-P-H1-FINAL-ONLY-V1'
read, raw, pin, file_sha = base.read, base.raw, base.pin, base.file_sha


def final_model_values(original):
    """Explicit successor fragment; caller writes it only in a fresh execution tree."""
    result = copy.deepcopy(original)
    if result.get('reasoning_budget_tokens') != 3072:
        raise ValueError('final correction preserves the 3072-token reasoning budget')
    result['decoding_params']['max_tokens'] = 16384
    result['answer_reserve_tokens'] = 13312
    result['geometry_basis'] = (
        'Final-only TF-GRPO consolidation correction: unchanged frozen model and '
        '49152-token resident context; completion cap 16384 with reasoning 3072 '
        'and answer reserve 13312. Earlier stage outputs and costs are imported '
        'with verified custody; no completed learning stage is rerun.')
    base.learning_decoding(result)
    return result


def documents(old_source, old_roster, *, final_import, runtime_identity,
              model_values, provenance, data_root, run_id=RUN):
    from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import learning_configuration, validate_registration
    from speech_aware_evidence_acquisition.rails.tfgrpo_final_import import load_import
    old = copy.deepcopy(old_source['batch'])
    if (old_source.get('schema') != 'saea-tfgrpo-learning-source-v1'
            or old.get('schema') != 'saea-tfgrpo-batch-v1'
            or old.get('prompt_sha256') != SOURCE_SHA256
            or old.get('upstream_commit') != UPSTREAM_COMMIT
            ):
        raise ValueError('original four-stage TF-GRPO source required')
    validate_experience(old['parent'], old['model_id'])
    if runtime_identity.get('model_fragment') != old['model_id']:
        raise ValueError('same frozen model fragment identity required')
    decoding = base.learning_decoding(model_values)
    if (decoding.get('max_tokens') != 16384
            or decoding.get('thinking_budget_tokens') != 3072
            or model_values.get('answer_reserve_tokens') != 13312):
        raise ValueError('explicit final fragment requires 16384 completion and 3072 reasoning')
    groups = old['groups']
    mode=old['common'].get('reward_mode',BINARY_REWARD)
    ids = [g['query_id'] for g in groups]
    if (not ids or len(set(ids)) != len(ids)
            or any(not q.startswith('slurp:train:') for q in ids)
            or any(not group_has_contrast(g['rollouts'],mode) for g in groups)):
        raise ValueError('unique original mixed TRAIN groups required')
    items = {x['sample_id']: x for x in old_roster['items']}
    expected_roster_ids={ids[0]} if 'final_import' in old else set(ids)
    if len(items) != len(old_roster['items']) or set(items) != expected_roster_ids:
        raise ValueError('original construction roster must cover all learned groups')
    # This call establishes scientific custody before deriving a fresh budget.
    _, custody = load_import(final_import, data_root=data_root, groups=groups,
        parent=old['parent'], common=old['common'], model_id=old['model_id'],
        policy_sha256=old['policy_sha256'], rollout_kind=old['rollout_kind'])
    anchor = ids[0]
    roster = copy.deepcopy(old_roster)
    roster.update(roster_id=run_id, count=1, items=[copy.deepcopy(items[anchor])],
        split_identity_hash=split_identity_hash(prefixed('slurp', [anchor])),
        selection='One original TRAIN anchor for final consolidation; all original learning groups remain bound in the batch.')
    roster_sha = __import__('hashlib').sha256(raw(roster)).hexdigest()
    plan = ExecutionPlan(run_id=run_id, attempt_id='learning-v1',
        execution_profile='bounded-discovery-probe', carrier_lock_key='slurp',
        split_role='discovery', split_identity_hash=roster['split_identity_hash'],
        planned_model_calls=1, planned_gpu_hours=8.0,
        planned_speech_audio_seconds=1, protocol_hash='0' * 64)
    registration = {
        'schema': 'saea-tfgrpo-runtime-binding-v1',
        'plan': {k: v for k, v in asdict(plan).items() if k != 'protocol_hash'},
        'model_id': old['model_id'], 'construction_roster_sha256': roster_sha,
        'contacts': [{'stage': 'batch_update', 'identity': 'batch', 'sample_id': anchor}],
        'decoding_params': decoding, 'max_context_bytes': 2000000,
        'runtime_identity_sha256': sha(runtime_identity),
    }
    registration['learning_config_sha256'] = sha(learning_configuration(registration))
    batch = copy.deepcopy(old)
    batch.pop('summary_import', None)
    batch['final_import'] = copy.deepcopy(final_import)
    batch['learning_config_sha256'] = registration['learning_config_sha256']
    batch['bounds'] = {**batch['bounds'], 'max_calls': 1, 'concurrency': 1}
    if custody.get('excluded_final_response',{}).get('failure_kind')=='experience-library-bound-exceeded':
        batch['common']['final_merge_limits']={k:batch['bounds'][k] for k in ('max_items','max_characters')}
        if 'final_import' in old:batch['common']['final_merge_policy']='bounded-synthesis-v1'
    registration['batch_sha256'] = sha(batch)
    validate_registration(registration, plan=plan, model_id=old['model_id'])
    source = {'schema': 'saea-tfgrpo-learning-source-v1', 'batch': batch,
              'provenance': copy.deepcopy(provenance)}
    return roster, source, registration, plan, custody


def prepare(repo, data, source_job, final_import, out, runtime_identity, *,
            job_name=NAME, run_id=RUN, service_session=None):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    if re.fullmatch(r'[a-z0-9][a-z0-9-]*', job_name) is None:
        raise ValueError('safe final job name required')
    if service_session is not None and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', service_session) is None:
        raise ValueError('safe service session required')
    if out.exists() or not out.resolve().is_relative_to((data / 'work/codex').resolve()):
        raise ValueError('fresh owned final job directory required')
    manifest = read(source_job / 'complete.json')
    for name in ('source.json', 'construction-roster.json'):
        if file_sha(source_job / name) != manifest['artifacts'][name]:
            raise ValueError('original prepared source artifact changed')
    old_source = read(source_job / 'source.json')
    old_roster = read(source_job / 'construction-roster.json')
    identity = read(runtime_identity)
    model_id = old_source['batch']['model_id']
    if base.deployment_identity(repo, model_id, identity) != identity:
        raise ValueError('new final fragment requires its exact current deployment identity')
    provenance = [pin(source_job / n, data, 'data') for n in
                  ('complete.json', 'source.json', 'construction-roster.json')]
    roster, source, reg, plan, custody = documents(old_source, old_roster,
        final_import=final_import, runtime_identity=identity,
        model_values=read(repo / f'configs/model/{model_id}.json'),
        provenance=provenance, data_root=data, run_id=run_id)
    out.mkdir(parents=True)
    artifacts = {'construction-roster.json': roster, 'source.json': source,
        'registration.json': reg, 'runtime-identity.json': identity}
    for name, doc in artifacts.items():
        publish_new(out / name, raw(doc))
    complete = {'schema': 'saea-tfgrpo-final-job-v1', 'job': job_name,
        'run_id': run_id, 'source_half': 'h1', 'evaluation_half': 'h2',
        'groups': len(source['batch']['groups']), 'runtime_rows': 1,
        'trajectories': sum(len(g['rollouts']) for g in source['batch']['groups']),
        'generation_calls': 1, 'fit_calls': 1, 'new_model_calls': 0,
        'new_tokenizer_calls': 0, 'batch_sha256': reg['batch_sha256'],
        'runtime_identity_sha256': sha(identity), 'inputs': provenance,
        'historical_custody_sha256': sha(custody),
        'artifacts': {n: file_sha(out / n) for n in artifacts},
        'development_queries_used_for_learning': False,
        'readiness': 'prepared-not-admitted'}
    if service_session is not None:
        complete['service_session'] = service_session
    publish_new(out / 'complete.json', raw(complete))
    return complete


def verify_job(repo, data, job):
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import read_pin, load_dynamic_source
    complete = read(job / 'complete.json')
    if complete.get('schema') == 'saea-learning-strategy-job-v1':
        from learning_strategy_job import verify_job as verify_strategy
        return verify_strategy(repo,data,job)
    if complete.get('schema') == 'saea-grounded-fusion-job-v1':
        from grounded_fusion_job import verify_job as verify_fusion
        return verify_fusion(repo,data,job)
    if complete.get('schema') == 'saea-grounded-shards-job-v1':
        from grounded_synthesis_job import verify_job as verify_grounded
        return verify_grounded(repo, data, job)
    if (complete.get('schema') != 'saea-tfgrpo-final-job-v1'
            or re.fullmatch(r'[a-z0-9][a-z0-9-]*', complete.get('job', '')) is None):
        raise ValueError('exact final job manifest required')
    required = {'construction-roster.json', 'source.json', 'registration.json', 'runtime-identity.json'}
    if set(complete['artifacts']) != required:
        raise ValueError('exact final job artifact set required')
    for name, digest in complete['artifacts'].items():
        if file_sha(job / name) != digest:
            raise ValueError('prepared final job changed')
    source = read(job / 'source.json')
    reg = read(job / 'registration.json')
    roster = read(job / 'construction-roster.json')
    identity = read(job / 'runtime-identity.json')
    if (sha(identity) != complete['runtime_identity_sha256']
            or base.deployment_identity(repo, reg['model_id'], identity) != identity):
        raise ValueError('final deployment identity differs')
    if source['provenance'] != complete['inputs'] or len(source['provenance']) != 3:
        raise ValueError('exact original source provenance required')
    manifest, old_source, old_roster = [read_pin(p, repo_root=repo, data_root=data)
                                      for p in source['provenance']]
    if any(manifest['artifacts'][name] != source['provenance'][i]['sha256']
           for i, name in ((1, 'source.json'), (2, 'construction-roster.json'))):
        raise ValueError('original source manifest differs')
    expected_roster, expected_source, expected_reg, plan, custody = documents(
        old_source, old_roster, final_import=source['batch']['final_import'],
        runtime_identity=identity, model_values=read(repo / f"configs/model/{reg['model_id']}.json"),
        provenance=source['provenance'], data_root=data, run_id=complete['run_id'])
    if (roster, source, reg) != (expected_roster, expected_source, expected_reg):
        raise ValueError('final job differs from exact verified source derivation')
    expected_counts = {'runtime_rows': 1, 'generation_calls': 1, 'fit_calls': 1,
        'groups': len(source['batch']['groups']),
        'trajectories': sum(len(g['rollouts']) for g in source['batch']['groups'])}
    if (any(complete.get(k) != v for k, v in expected_counts.items())
            or complete['batch_sha256'] != reg['batch_sha256']
            or complete['historical_custody_sha256'] != sha(custody)):
        raise ValueError('final job counts or historical custody differ')
    load_dynamic_source({'prepared_training_feedback': {'schema': 'saea-prepared-tfgrpo-dynamic-v1',
        'source': pin(job / 'source.json', data, 'data')}, 'tfgrpo_learning_registration': reg},
        plan=plan, repo_root=repo, data_root=data,
        sample_ids=tuple(x['sample_id'] for x in roster['items']),
        roster_sha256=file_sha(job / 'construction-roster.json'))
    return complete, reg, roster


def register(repo,data,job):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment();require_clean_study_repo(repo)
    complete,reg,roster=verify_job(repo,data,job)
    name=complete.get('job',NAME);run_id=complete.get('run_id',RUN)
    from tfrl_current_generation import SESSION
    service_session=complete.get('service_session',SESSION)
    names=[run_id,run_id+'-FIT'];ledger=repo/'docs/exposure-ledger.md'
    if any('| '+rid+' |' in ledger.read_text() or (data/'attempts'/rid).exists() for rid in names):raise ValueError('existing learning registration or spend')
    model_id=reg['model_id'];model=read(repo/f'configs/model/{model_id}.json')
    if reg['batch_sha256']!=complete['batch_sha256'] or len(reg['contacts'])!=complete['generation_calls']:
        raise ValueError('job registration and source budget differ')
    roster_rel=f'docs/receipts/rosters/{name}.json'
    baseline=read(repo/'configs/baseline/system-group-update-text-v1.json');baseline.pop('whole_roster_wave_barrier',None)
    baseline['what_this_is']='Final-only TF-GRPO consolidation of verified H1 group operations; one actual contact, all historical costs retained.'
    fusion = complete.get('schema') == 'saea-grounded-fusion-job-v1'
    grounded = complete.get('schema') == 'saea-grounded-shards-job-v1'
    incremental = complete.get('schema') == 'saea-learning-strategy-job-v1'
    reflexion = incremental and complete.get('strategy') == 'reflexion_offline_transfer_v1'
    if incremental:
        baseline['what_this_is']='Offline ACE local reflection and incremental curation over original TRAIN evidence; each realized prompt requires native FIT; no causal attribution or evaluated benefit claimed.'
    if reflexion:
        baseline['what_this_is']='Offline Reflexion transfer over historical TRAIN evidence; one reflection per group, deterministic bounded FIFO, no actor retries or curator; candidate requires independent review and evaluation.'
    if grounded:
        baseline['what_this_is']='Evidence-grounded H1 synthesis shards; intermediate artifacts only, complete historical costs retained.'
    if fusion:
        baseline['what_this_is']='Evidence-grounded final fusion with explicit rule support and proposal dispositions; candidate requires semantic review and development evaluation.'
    baseline.pop('phase_material',None)
    from speech_aware_evidence_acquisition.rails.prepared_text import PREPARED_TEXT_INSTRUCTION
    baseline['task_instruction']=PREPARED_TEXT_INSTRUCTION
    baseline['serving_slot_tokens']=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
    baseline['serving_slots']=read(repo/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json')['serving_slots']
    exp=read(repo/'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json')
    for k in ('population','stage3_measurement','prepared_phase_supply','native_episode','knowledge_stores','live_phase_fit','native_contact_fit'):
        exp.pop(k,None)
    exp.update(experiment_id=run_id,purpose=baseline['what_this_is'],execution_plan={k:reg['plan'][k] for k in ('execution_profile','split_role','planned_model_calls','planned_gpu_hours','planned_speech_audio_seconds')},
        tfgrpo_learning_registration=reg,prepared_training_feedback={'schema':'saea-prepared-tfgrpo-dynamic-v1','source':pin(job/'source.json',data,'data')},
        artifact_result={'schema':'saea-artifact-result-binding-v1','role':'training-feedback-proposal','construction_roster':{'repo_relpath':roster_rel,'sha256':file_sha(job/'construction-roster.json')}},
        tfgrpo_contact_fit={'run_id':run_id+'-FIT','attempt_id':'learning-fit-v1','contact_cap':complete['fit_calls']},
        stage3_algorithm={'id':'tfgrpo-system-memory-disjoint-v1','operators_used':['credit_assignment'],'note':'One final consolidation of verified completed four-stage inputs; shared-P adaptation; candidate only.'})
    cell,=exp['cells'];cell.update(cell_id=name,run_id=run_id,attempt_id='learning-v1',baseline_fragment=name,
        allowed_execution_stages=['check','worker','campaign'],readiness={'state':'held','requirements':[],'reason':'Registered exact learning source; await shared live runtime preflight and lease.'})
    if grounded:
        exp['prepared_training_feedback']['schema']='saea-prepared-grounded-shards-v1'
        exp['stage3_algorithm']['note']='Grounded intermediate shards over verified H1 evidence; no final candidate or held-out gain claimed.'
    if fusion:
        exp['prepared_training_feedback']['schema']='saea-prepared-grounded-fusion-v1'
        exp['stage3_algorithm']['note']='Final H1 fusion over finalized shards and original scored predictions; candidate not accepted for deployment.'
    if incremental:
        exp['prepared_training_feedback']['schema']='saea-prepared-ace-dynamic-v1'
        exp['stage3_algorithm']={'id':'ace-grow-refine-v1','operators_used':['update'],
            'note':complete['strategy']+': offline local Reflector/Curator, finite observed-prefix dynamic prompts; later stages are not fitted or generated before their predecessors are observed.'}
    if reflexion:
        exp['prepared_training_feedback']['schema']='saea-prepared-reflexion-dynamic-v1'
        exp['stage3_algorithm']={'id':'reflexion-offline-transfer-v1','operators_used':['update'],
            'note':complete['strategy']+': capacity-controlled offline transfer, one observed-prefix reflection per TRAIN group; not original same-task retries or equal-budget ACE.'}
    exp['populations'][0]['ids']=[x['sample_id'] for x in roster['items']]
    exp['service_session']=service_session
    for rel,doc in [(roster_rel,roster),(f'configs/baseline/{name}.json',baseline),(f'configs/experiment/{name}.json',exp)]:
        path=repo/rel
        if path.exists():raise ValueError('learning configuration already exists')
        path.parent.mkdir(parents=True,exist_ok=True);publish_new(path,raw(doc))
    cfg=compose(repo,model_id,cell['dataset_fragment'],name,name);ph=protocol_hash(cfg.config_hash,cfg.values['arm'],roster['split_identity_hash'])
    package=repo/'docs/readiness'/name;package.mkdir(parents=True)
    rows=[]
    for kind,run,attempt in [('generation',run_id,'learning-v1'),('fit',run_id+'-FIT','learning-fit-v1')]:
        plan=ExecutionPlan(**{**reg['plan'],'run_id':run,'attempt_id':attempt,'protocol_hash':ph}).validate(source_root=repo)
        doc={'schema':'saea-tfgrpo-registered-plan-v1','execution_plan':asdict(plan),'job_manifest':pin(job/'complete.json',data,'data'),
             'config_sha256':cfg.config_hash,'authority':base.AUTH,'actual_audio_seconds_expected':0,'readiness':'held'}
        path=package/(kind+'-plan.json');publish_new(path,raw(doc))
        rows.append(['2026-11-02',run,'bounded-discovery-probe',('Offline Reflexion transfer ' if reflexion else 'Offline ACE incremental ' if incremental else 'Grounded H1 final fusion ' if fusion else 'Grounded H1 intermediate shards ' if grounded else 'TF-GRPO H1 final-only consolidation ')+kind,model_id,'slurp TRAIN original source groups','discovery',roster['split_identity_hash'],'no',str(plan.planned_model_calls),str(plan.planned_speech_audio_seconds),str(plan.planned_gpu_hours),'All source rollout costs retained',str(path.relative_to(repo)),f'Plan SHA256 {file_sha(path)}; owner-authorized under {base.AUTH}; no H2 feedback in learning; held; text only, actual audio zero'])
    ledger.write_text(ledger.read_text()+'\n## Fixed-P TF-GRPO learning\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n')
    return {'job':name,'rows':len(roster['items']),'learned_groups':complete['groups'],'calls':complete['generation_calls'],'fit_calls':complete['fit_calls'],'readiness':'held','new_model_calls':0}


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
    assert_execution_environment();require_clean_study_repo(repo)
    complete,reg,roster=verify_job(repo,data,job)
    name=complete.get('job',NAME);run_id=complete.get('run_id',RUN)
    from tfrl_current_generation import SESSION
    service_session=complete.get('service_session',SESSION)
    pkg=repo/'docs/readiness'/name
    exp=read(repo/f'configs/experiment/{name}.json');cell,=exp['cells']
    if cell['readiness']['state']!='held' or exp['tfgrpo_learning_registration']!=reg:
        raise ValueError('exact held registration required')
    resident=find_resident_session(data)
    if resident is None or resident[1]['session_id']!=service_session:raise ValueError('expected resident absent')
    assert_identity_matches(resident[1],read(job/'runtime-identity.json'))
    receipt=resolve_runtime_receipt_path(repo,read_fragment(repo,'model',reg['model_id']))
    receipt_doc=read(receipt)
    for instance in receipt_doc['deployment']['instances']:
        session=SessionReceipt.from_json(read(data/'serving'/instance['instance_id']/'session-receipt.json'))
        verify_vllm_session(session,receipt_doc)
    runs=[run_id,run_id+'-FIT'];decision_id=run_id+'-ADMISSION';decision_rel='coordination/decisions/'+decision_id+'.json'
    if any((data/'attempts'/rid).exists() for rid in runs):raise ValueError('existing learning spend')
    draft={'schema':'saea-campaign-decision-v1','status':'DRAFT-NOT-MINTED','decision_id':decision_id,
        'eligible_run_ids':runs,'eligible_attempt_ids':['learning-v1','learning-fit-v1'],'authorization_basis':base.AUTH,
        'owner':'root','decision':('Execute bounded offline Reflexion single-stage reflection with deterministic FIFO only after predecessor observation and native FIT; all TRAIN costs retained; independent review required.' if complete.get('strategy')=='reflexion_offline_transfer_v1' else 'Execute bounded offline ACE Reflector/Curator stages only after predecessor observation and actual native FIT; all TRAIN source costs retained; candidate requires evaluation.' if complete.get('schema')=='saea-learning-strategy-job-v1' else 'Execute registered grounded intermediate shards with native FIT and complete source costs; no final candidate.' if complete.get('schema')=='saea-grounded-shards-job-v1' else 'Execute one final consolidation of verified completed H1 stages through shared TF-GRPO runtime; all learned groups and historical costs retained; no H2 feedback.'),
        'antecedents':{'owner':{'path':base.AUTH,'sha256':file_sha(repo/base.AUTH)},
            'runtime':{'path':str(receipt.relative_to(repo)),'sha256':file_sha(receipt)}},
        'source_manifest':pin(job/'complete.json',data,'data'),'service':{'expected_session_id':service_session},
        'budget':{k:read(pkg/(k+'-plan.json'))['execution_plan'] for k in ('generation','fit')}}
    publish_new(pkg/'root-admission-draft-v1.json',raw(draft))
    family=SimpleNamespace(PACKAGE=str(pkg.relative_to(repo)),DECISION_ID=decision_id,DECISION_RELPATH=decision_rel,eligible_run_ids=lambda:runs)
    result=mint(str(repo),str(data),'Codex',False,family=family,tool='scripts/analysis/tfgrpo_learning_job.py')
    apply_decision_and_readiness(repo_root=repo,data_root=data,experiment=name,cell_id=name,
        decision=(decision_rel,decision_id),set_ready=True)
    cfg=compose(repo,reg['model_id'],cell['dataset_fragment'],name,name)
    ph=protocol_hash(cfg.config_hash,cfg.values['arm'],roster['split_identity_hash'])
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    for kind in ('generation','fit'):
        path=pkg/(kind+'-plan.json');old=file_sha(path);doc=read(path)
        doc['execution_plan']['protocol_hash']=ph;doc['config_sha256']=cfg.config_hash;doc['readiness']='admitted-pending-clean-commit'
        path.write_bytes(raw(doc));text=text.replace(old,file_sha(path))
    ledger.write_text(text)
    return {**result,'new_model_calls':0,'remaining':['commit admission','shared preflight','generation lease after predecessor release']}


def run(repo,data,job,out,after):
    """One detached dispatch; campaign alone acquires and releases the lease."""
    import time
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells,declared_readiness,run_campaign
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from run_real_engine_candidate import assert_campaign_success
    assert_execution_environment();commit=require_clean_study_repo(repo)
    complete,reg,roster=verify_job(repo,data,job)
    name=complete.get('job',NAME);run_id=complete.get('run_id',RUN)
    from tfrl_current_generation import SESSION
    service_session=complete.get('service_session',SESSION)
    base.require_learning_deployment(repo,reg['model_id'])
    source=read(job/'source.json')
    predecessor=('SAEA-SLURP-TFGRPO-RENDER-H2-THREE-ARM-V1' if source['batch']['common'].get('reward_mode',BINARY_REWARD)!=BINARY_REWARD else 'SAEA-SLURP-TFRL-HARDSET-A0-OBSERVATION-R1-V1')
    base.verify_release(after,data,expected_run=predecessor)
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned output required; no blind retry')
    cells=load_campaign_cells(repo,name);cell,=cells
    if declared_readiness(cell,data).state!='ready':raise ValueError('learning job remains held')
    for rid,aid in ((run_id,'learning-v1'),(run_id+'-FIT','learning-fit-v1')):
        if (data/'attempts'/rid/(aid+'.json')).exists():raise ValueError('existing attempt; reconcile before resume')
    out.mkdir(parents=True);publish_new(out/'started.json',raw({'study_commit':commit,'job':name,'predecessor_release_sha256':file_sha(after)}))
    try:
        runtime=build_flight_runtime(repo,data,model=reg['model_id'],dataset=cell.document['dataset_fragment'],
            baseline=name,experiment=name,run_id=run_id,attempt_id='learning-v1',runtime_root=data/'work/scratch/runtime-root',
            lock_path=repo.parents[1]/'docs/datasets.lock.json')
        result=run_campaign([runtime],cells=cells,repo_root=repo,data_root=data,owner='Codex-TFGRPO',expected_session_id=service_session)
        publish_new(out/'campaign.json',raw(result));assert_campaign_success(result,run_id)
        publish_new(out/'complete.json',raw({'job':name,'groups':complete['groups'],'result':result,'candidate_accepted':False}))
        return {'completed':True,'candidate_accepted':False}
    except BaseException as error:
        publish_new(out/'failed.json',raw({'type':type(error).__name__,'error':str(error),'automatic_retry':False,
            'preserve':'learning receipts, pending contact intents, FIT and generation costs'}));raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'register', 'verify', 'admit', 'run'])
    for name in ('repo', 'data'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-job', 'final-import', 'out', 'runtime-identity', 'job', 'after'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--job-name', default=NAME)
    parser.add_argument('--run-id', default=RUN)
    parser.add_argument('--service-session')
    args = vars(parser.parse_args())
    mode = args.pop('mode')
    if mode == 'prepare':
        args['final_import'] = {k: v for k, v in pin(args['final_import'], args['data'], 'data').items() if k != 'root'}
        result = prepare(**{k: args[k] for k in ('repo', 'data', 'source_job', 'final_import', 'out', 'runtime_identity', 'job_name', 'run_id', 'service_session')})
    elif mode == 'verify':
        result = verify_job(args['repo'], args['data'], args['job'])[0]
    elif mode == 'run':
        result = run(**{k: args[k] for k in ('repo', 'data', 'job', 'out', 'after')})
    else:
        result = globals()[mode](args['repo'], args['data'], args['job'])
    print(__import__('json').dumps(result, sort_keys=True))
