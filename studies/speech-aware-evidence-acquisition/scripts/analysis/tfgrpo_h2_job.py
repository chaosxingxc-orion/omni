"""Stage and register full-H2 paired development through existing flight tools.

CPU-only commands never admit or dispatch. Actual candidate sealing precedes
template construction; per-request native FIT remains authoritative at dispatch.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
import tfgrpo_h2_paired as paired
from tfgrpo_h2_paired import NAME, RUN, ROSTER, AUTH
from tfrl_current_generation import MODEL, SESSION, PROFILE, RECEIPT, LOCK, LOCK_SHA
from tfrl_hardset_a0_registration import pin, pinned, scientific_hash
from tfrl_hardset_a0_workload import read, raw, sha, confined
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption

TEMPLATE = 'probes/tfgrpo-fixed-p-h1-experience-v1.txt'
ESTIMATE_SOURCE = 'configs/baseline/slurp-tfrl-odd40-i1-h2-v1.json'
KIND = 'tfgrpo-full-h2-paired'



RUNTIME_BINDING = 'configs/evaluation/tfgrpo-h2-runtime-v1.json'


def runtime_binding(repo, evaluation=None):
    """An explicit successor binding preserves historical default experiments."""
    path=Path(repo)/RUNTIME_BINDING
    if evaluation is None and not path.exists():
        return {'model_fragment':MODEL,'service_session':SESSION,'profile':PROFILE,'runtime_receipt':RECEIPT}
    spec=read(path) if evaluation is None else paired.evaluation_identity(evaluation)['runtime']
    if set(spec)!={'model_fragment','service_session','profile','runtime_receipt'} or (evaluation is None and spec['model_fragment']!=MODEL):
        raise ValueError('exact frozen-model H2 runtime binding required')
    for key in ('model_fragment','service_session','profile'):
        value=spec[key]
        if not isinstance(value,str) or not value or Path(value).name!=value or value in ('.','..'):
            raise ValueError('confined runtime identifier required')
    model=read(confined(repo,f"configs/model/{spec['model_fragment']}.json"))
    profile=read(confined(repo,f"configs/deployment/{spec['profile']}.json"))
    if model['runtime_receipt']!=spec['runtime_receipt']:
        raise ValueError('model/profile runtime binding differs')
    if profile['model_fragment']!=spec['model_fragment']:
        other=Path(repo)/f"configs/model/{profile['model_fragment']}.json"
        if not other.is_file():raise ValueError('profile model differs')
        deployed=read(other)
        protocol=lambda doc:{k:v for k,v in doc.items() if k!='geometry_basis'}
        if protocol(model)!=protocol(deployed):raise ValueError('profile model protocol differs')
    confined(Path(repo)/'docs',spec['runtime_receipt'])
    return spec


def require_unspent(repo, data, evaluation=None):
    RUN=paired.evaluation_identity(evaluation)['run_id']
    ledger = (repo/'docs/exposure-ledger.md').read_text()
    for run in (RUN, RUN+'-FIT', RUN+'-READ'):
        if '| '+run+' |' in ledger or any((data/k/run).exists() for k in ('attempts','runs')):
            raise ValueError('existing H2 registration or spend; inspect before continuation')


def fresh_output(data, out):
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output required')


def measure_templates(repo, data, params, evaluation=None):
    """CPU tokenization of exact templates, not a full-context FIT claim."""
    from transformers import AutoTokenizer
    binding=runtime_binding(repo,evaluation)
    MODEL=binding['model_fragment']
    session=binding['service_session'];profile=binding['profile'];receipt_rel=binding['runtime_receipt']
    receipt_path = repo/'docs'/receipt_rel
    receipt = read(receipt_path)
    model_dir = Path(receipt['model']['model_dir']).resolve()
    if not model_dir.is_relative_to(data.resolve()/'models'):
        raise ValueError('owned model tokenizer required')
    expected = {entry['path']: entry['sha256'] for entry in receipt['model']['files']}
    token_files = {}
    for name in ('config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','added_tokens.json','special_tokens_map.json'):
        path = model_dir/name
        if path.exists():
            rel = str(path.relative_to(data.resolve()))
            if expected.get(rel) != sha(path):
                raise ValueError('tokenizer file differs from resident receipt: '+name)
            token_files[rel] = sha(path)
    if not {'tokenizer_config.json','vocab.json','merges.txt'} <= {Path(p).name for p in token_files}:
        raise ValueError('complete resident BPE tokenizer custody required')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    counts = {k: len(tokenizer.encode(consumption.checked_pin(repo/'configs',params[k]).read_text(),add_special_tokens=False))
              for k in ('base_template','template')}
    estimate = copy.deepcopy(read(repo/ESTIMATE_SOURCE)['native_episode']['consumption_module']['supply_frame']['fit_estimate'])
    model = read(repo/f'configs/model/{MODEL}.json')
    if evaluation is not None or (repo/RUNTIME_BINDING).exists():
        estimate['slot_tokens']=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
    if (estimate['slot_tokens'] != int(model['server_args'][model['server_args'].index('--max-model-len')+1])
            or estimate['completion_ceiling'] != model['decoding_params']['max_tokens']):
        raise ValueError('inherited compile estimate differs from resident context geometry')
    return {'schema':'saea-tfgrpo-template-measurement-v1','method':'resident-local-tokenizer-add-special-tokens-false',
        'base_template':params['base_template'],'template':params['template'],'token_counts':counts,
        'tokenizer_files':token_files,'runtime_receipt':pin(repo,'docs/'+receipt_rel),
        'fit_estimate':estimate,'fit_estimate_source':pin(repo,ESTIMATE_SOURCE),
        'runtime_binding':binding,'full_prompt_measured':False,'actual_request_fit':'native_contact_fit at every dispatch',
        'new_model_calls':0,'new_tokenizer_endpoint_calls':0}


def stage(repo, data, candidate, manifest, out, evaluation=None):
    identity=paired.evaluation_identity(evaluation)
    NAME,RUN,TEMPLATE=identity['name'],identity['run_id'],identity['template']
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    binding=runtime_binding(repo,evaluation)
    MODEL=binding['model_fragment']
    session=binding['service_session'];profile=binding['profile'];receipt_rel=binding['runtime_receipt']
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment(); head = require_clean_study_repo(repo)
    require_unspent(repo,data,evaluation); fresh_output(data,out)
    targets = [repo/f'configs/{kind}/{NAME}.json' for kind in ('baseline','experiment')]
    if any(p.exists() for p in targets) or (repo/'configs'/TEMPLATE).exists():
        raise ValueError('H2 derivation already exists')
    roster = read(repo/ROSTER); ids = [r['sample_id'] for r in roster['items']]
    base = read(repo/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json')
    params = consumption.prepare(candidate_path=candidate,source_manifest=manifest,repo=repo,data=data,
        base_template=base['phase_material']['solver']['instruction_template'],template=TEMPLATE,
        consumer_ids=ids,model_id=MODEL,**({'candidate_release':evaluation['candidate_release']} if evaluation is not None else {}))
    measurement = measure_templates(repo,data,params,evaluation)
    derived = paired.derive(repo,data,params,fit_estimate=measurement['fit_estimate'],
        base_template_tokens=measurement['token_counts']['base_template'],learned_template_tokens=measurement['token_counts']['template'],evaluation=evaluation)
    exp = derived['experiment']; cell, = exp['cells']
    cell['model_fragment']=MODEL
    cell['allowed_execution_stages'] = ['check','prepare-supply','worker','campaign']
    exp['service_session'] = session
    out.mkdir(parents=True)
    for path, doc in zip(targets,(derived['baseline'],exp)):
        path.parent.mkdir(parents=True,exist_ok=True); publish_new(path,raw(doc))
    publish_new(out/'template-measurement.json',raw(measurement))
    record = {'schema':'saea-tfgrpo-h2-stage-v1','rows':312,'run_id':RUN,'source_commit':head,
        'params':params,'measurement':pin(data,str((out/'template-measurement.json').relative_to(data))),
        'evaluation':derived['evaluation'],'output_hashes':{str(p.relative_to(repo)):sha(p) for p in targets},
        'new_model_calls':0,'new_tokenizer_endpoint_calls':0,'generation_authorized':False,'readiness':'held'}
    if evaluation is not None:record['evaluation_config']=copy.deepcopy(evaluation)
    publish_new(out/'complete.json',raw(record)); return record


def prepare_supply(repo, data, stage_record, out):
    evaluation=read(stage_record).get('evaluation_config')
    identity=paired.evaluation_identity(evaluation); NAME,RUN=identity['name'],identity['run_id']
    MODEL=runtime_binding(repo,evaluation)['model_fragment']
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from tfrl_odd40_cells import verify_output
    assert_execution_environment(); require_clean_study_repo(repo); require_unspent(repo,data,evaluation); fresh_output(data,out)
    staged = verify_stage(repo,data,stage_record)
    exp = read(repo/f'configs/experiment/{NAME}.json'); cell, = exp['cells']
    runtime = build_flight_runtime(repo,data,model=MODEL,dataset=cell['dataset_fragment'],baseline=NAME,
        experiment=NAME,run_id=RUN,attempt_id='generation-v1',runtime_root=data/'work/scratch/runtime-root',
        lock_path=repo.parents[1]/'docs/datasets.lock.json')
    target = data/'prepared'/RUN/'phase-supply-v1.json'
    runtime.prepare_phase_supply(target,max_workers=16)
    verify_output(read(target)['output'],exp['populations'][0]['ids'])
    out.mkdir(parents=True)
    record = {'schema':'saea-tfgrpo-h2-first-p-supply-v1','rows':312,'run_id':RUN,
        'stage_sha256':sha(stage_record),'supply':pin(data,str(target.relative_to(data))),
        'new_model_calls':0,'new_tokenizer_endpoint_calls':0,'generation_authorized':False}
    publish_new(out/'complete.json',raw(record)); return record


def verify_stage(repo, data, stage_record):
    staged = read(stage_record)
    evaluation=staged.get('evaluation_config')
    identity=paired.evaluation_identity(evaluation); RUN=identity['run_id']
    MODEL=runtime_binding(repo,evaluation)['model_fragment']
    if staged['schema'] != 'saea-tfgrpo-h2-stage-v1' or staged['rows'] != 312 or staged['run_id'] != RUN:
        raise ValueError('exact H2 stage required')
    for rel,digest in staged['output_hashes'].items():
        if sha(confined(repo,rel)) != digest: raise ValueError('H2 staged configuration changed')
    params = staged['params']
    if evaluation is not None and (params.get('candidate_release') != evaluation['candidate_release']
            or params['template']['relpath'] != identity['template']):
        raise ValueError('staged evaluation release or template differs')
    consumption.verify(params,repo=repo,data=data,model_id=MODEL,consumer_ids=[r['sample_id'] for r in read(repo/ROSTER)['items']])
    measured = pinned(data,staged['measurement'])
    if measured != measure_templates(repo,data,params,evaluation): raise ValueError('template measurement changed')
    return staged


def validate(repo,data,name,spec,exp,view):
    evaluation=spec.get('evaluation_config')
    identity=paired.evaluation_identity(evaluation); NAME,RUN=identity['name'],identity['run_id']
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    binding=runtime_binding(repo,evaluation)
    MODEL=binding['model_fragment']
    session=binding['service_session'];profile=binding['profile'];receipt_rel=binding['runtime_receipt']
    from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
    from tfrl_odd40_cells import verify_output
    cell, = exp['cells']; pop, = exp['populations']; summary=exp['population']
    roster=pinned(repo,spec['roster']); ids=[r['sample_id'] for r in roster['items']]
    if (name != NAME or spec['run_id'] != RUN or spec['attempt_id'] != 'generation-v1'
            or cell['run_id'] != RUN or cell['attempt_id'] != spec['attempt_id']
            or cell['model_fragment'] != MODEL or cell['baseline_fragment'] != NAME
            or spec['fit_run_id'] != RUN+'-FIT' or spec['fit_attempt_id'] != 'fit-v1'
            or spec['profile'] != profile or spec['runtime_receipt'] != receipt_rel
            or spec['service_session'] != session or exp['service_session'] != session):
        raise ValueError('exact paired H2 runtime identity required')
    if (spec['roster']['relpath'] != ROSTER or roster != read(repo/ROSTER)
            or len(ids) != 312 or len(set(ids)) != 312 or spec['rows'] != 312 or summary['count'] != 312
            or pop['ids'] != ids or summary['roster_receipt'] != ROSTER
            or summary['dataset_fragment'] != cell['dataset_fragment'] or pop['dataset_fragment'] != cell['dataset_fragment']
            or view['count'] != 625 or len(view['items']) != 625 or pinned(repo,spec['dataset']) != view
            or len({r['sample_id'] for r in view['items']}) != 625
            or not set(ids) <= {r['sample_id'] for r in view['items']}
            or len({split_identity_hash(prefixed('slurp',ids)),roster['split_identity_hash'],spec['split_identity_hash'],summary['split_identity_hash']}) != 1):
        raise ValueError('exact full H2 within fixed625 dataset required')
    byid={r['sample_id']:r for r in view['items']}
    for row in roster['items']:
        actual=byid[row['sample_id']]
        if (not row['sample_id'].startswith('slurp:train:') or '/slurp_real/' not in actual['media_relpath']
                or len(actual['audio_sha256']) != 64 or actual['audio_seconds'] != row['audio_seconds']):
            raise ValueError('real audio custody changed')
    base=read(repo/f'configs/baseline/{NAME}.json'); fit=base['native_contact_fit']
    if (scientific_hash(base) != spec['scientific_baseline_sha256']
            or not native_contact_custody_enabled({**base,'transport':'vllm','carrier_lock_key':'slurp'})
            or fit != {'run_id':RUN+'-FIT','attempt_id':'fit-v1','contact_cap':936,'max_speech_packets':1}
            or exp['execution_plan']['planned_model_calls'] != 1872
            or exp['execution_plan']['planned_speech_audio_seconds'] < math.ceil(6*sum(r['audio_seconds'] for r in roster['items']))
            or exp['stage3_algorithm']['id'] != ('ace-grow-refine-v1' if evaluation is not None else 'tfgrpo-semantic-advantage-v1')
            or exp['stage3_algorithm']['operators_used'] != (['context_application'] if evaluation is not None else ['context_application','rollout','scoring'])
            or exp['stage3_measurement']['claims_benefit'] is not False):
        raise ValueError('registered paired program or budget changed')
    if evaluation is not None:
        components=base['native_episode']['consumption_module']['components']
        matches=[c['params'] for c in components if c['component_id']==consumption.COMPONENT and c['state']=='on']
        if len(matches)!=1 or matches[0].get('candidate_release')!=evaluation['candidate_release'] or matches[0]['template']['relpath']!=identity['template']:
            raise ValueError('registered candidate release differs')
    consumption.verify_for_flight(base,repo=repo,data=data,model_id=MODEL,consumer_ids=ids)
    verify_output(pinned(data,base['prepared_phase_supply'])['output'],ids)
    if spec['gain_lock'] != {'relpath':LOCK,'sha256':LOCK_SHA} or pinned(repo,spec['gain_lock'])['status'] != 'locked_full_train_fold_b':
        raise ValueError('locked first-system prerequisite differs')
    return 312


def register(repo,data,stage_record,prepared):
    evaluation=read(stage_record).get('evaluation_config')
    identity=paired.evaluation_identity(evaluation); NAME,RUN=identity['name'],identity['run_id']
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment,ExecutionPlan
    binding=runtime_binding(repo,evaluation)
    MODEL=binding['model_fragment']
    session=binding['service_session'];profile=binding['profile'];receipt_rel=binding['runtime_receipt']
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment(); head=require_clean_study_repo(repo); require_unspent(repo,data,evaluation)
    staged=verify_stage(repo,data,stage_record); prep=read(prepared)
    if (prep['schema'] != 'saea-tfgrpo-h2-first-p-supply-v1' or prep['rows'] != 312
            or prep['run_id'] != RUN or prep['stage_sha256'] != sha(stage_record)
            or prep['new_model_calls'] != 0 or prep['new_tokenizer_endpoint_calls'] != 0):
        raise ValueError('exact CPU first-P preparation required')
    bp=repo/f'configs/baseline/{NAME}.json'; ep=repo/f'configs/experiment/{NAME}.json'
    base=read(bp); exp=read(ep); cell,=exp['cells']; pkg=repo/'docs/readiness'/NAME
    if pkg.exists() or cell['readiness']['state'] != 'held' or 'prepared_phase_supply' in base:
        raise ValueError('fresh held H2 registration required')
    base['prepared_phase_supply']=prep['supply']
    cell['readiness']['requirements']=[{'kind':'preparation',**prep['supply']}]
    bp.write_bytes(raw(base)); ep.write_bytes(raw(exp))
    cfg=compose(repo,MODEL,cell['dataset_fragment'],NAME,NAME); split=exp['population']['split_identity_hash']
    ph=protocol_hash(cfg.config_hash,cfg.values['arm'],split)
    identity={'cell':NAME,'experiment':NAME,'run_id':RUN,'attempt_id':'generation-v1','fit_run_id':RUN+'-FIT',
        'fit_attempt_id':'fit-v1','rows':312,'composed_config_hash':cfg.config_hash,'protocol_hash':ph,'split_identity_hash':split}
    spec={**identity,'workload_kind':('learning-full-h2-paired' if evaluation is not None else KIND),'model_fragment':MODEL,'baseline':NAME,'service_session':session,
        'profile':profile,'runtime_receipt':receipt_rel,'authority':AUTH,'decision_id':NAME+'-owner-admission',
        'decision_relpath':f'coordination/{NAME}/root-admission-v1.json','roster':pin(repo,ROSTER),
        'dataset':pin(repo,f"configs/dataset/{cell['dataset_fragment']}.json"),'gain_lock':pin(repo,LOCK),
        'scientific_baseline_sha256':scientific_hash(base),'stage_sha256':sha(stage_record)}
    if evaluation is not None:spec['evaluation_config']=copy.deepcopy(evaluation)
    validate(repo,data,NAME,spec,exp,read(repo/spec['dataset']['relpath']))
    protocol={'schema':'saea-tfgrpo-h2-paired-generation-v1','cells':{'arm':identity},'profile':profile,
        'roster':spec['roster'],'gain_lock':spec['gain_lock'],'prepared_phase_supply':base['prepared_phase_supply'],
        'evaluation':staged['evaluation'],'template_measurement':staged['measurement'],'claims_benefit':False}
    docs={'candidate.json':spec,'protocol-registered-v1.json':protocol}
    protocol_pin={'relpath':str((pkg/'protocol-registered-v1.json').relative_to(repo)),'sha256':hashlib.sha256(raw(protocol)).hexdigest()}
    rows=[]
    for kind,rid,aid,calls in [('generation',RUN,'generation-v1',1872),('fit',RUN+'-FIT','fit-v1',936)]:
        budget={**exp['execution_plan'],'planned_model_calls':calls}
        plan=ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',split_identity_hash=split,protocol_hash=ph,**budget)
        plan.validate(source_root=repo)
        doc={'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':NAME,'experiment':NAME,
            'composed_config_hash':cfg.config_hash,'expected_session_id':session,'execution_plan':asdict(plan),
            'prepared_phase_supply':base['prepared_phase_supply'],'protocol_document':protocol_pin,'status':'held'}
        rel=f'{kind}-plan-registered-v1.json'; docs[rel]=doc
        rows.append(['2026-11-01',rid,'bounded-discovery-probe','Full H2 paired development '+kind,MODEL,'slurp TRAIN H2 fixed312',
            'discovery',split,'no',str(calls),str(budget['planned_speech_audio_seconds']),str(budget['planned_gpu_hours']),
            'H1 learning and historical H2 exposure retained',str((pkg/rel).relative_to(repo)),
            f'Plan SHA256 {hashlib.sha256(raw(doc)).hexdigest()}; root countersigned under {AUTH}; no test tuning; held'])
    docs['analysis-recipe-v1.json']={'schema':'saea-tfgrpo-h2-paired-read-v1','rows':312,'generation_run_id':RUN,
        'attempt_id':'generation-v1','roster':spec['roster'],'arms':['learned','lock'],'delivered_arm':'lock',
        'source_verification':'verify_bundle requires FINALIZED complete312; no filtering',
        'render':'stage3_rollout_score.render_supply_variant_first_visit',
        'score':'stage3_rollout_score.score_supply_variant_copies; official full-roster scoring per arm',
        'evaluation':staged['evaluation'],'new_model_calls':0,'candidate_accepted':False,
        'report':['SLU-F1','Intent','Scenario','Span','paired uncertainty','per-label regressions','tokens','latency','allocated card-hours']}
    pkg.mkdir(parents=True)
    for rel,doc in docs.items(): publish_new(pkg/rel,raw(doc))
    ledger=repo/'docs/exposure-ledger.md'
    label='TF-GRPO' if evaluation is None else 'ACE frozen candidate'
    ledger.write_text(ledger.read_text()+'\n## Full H2 '+label+' paired development\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n')
    publish_new(repo/'docs/readiness'/f'{NAME}-preregistration.md',b'# Full H2 paired development\n\nFixed312 H2; one P and two Solver arms, learned and lock. Shared campaign delivers lock; officially score both complete arms separately. No H2 feedback used in H1 learning. Candidate is not accepted by generation. Gate: +1 percentage point SLU-F1 with no Intent decrease; report paired uncertainty and per-label regressions. 936 normal generation contacts,1872 fallback cap,936 FIT. Commit registration, use shared output-preserving admission/repin, commit, then detached shared dispatch and native recovery.\n')
    return {'cell':NAME,'rows':312,'source_commit':head,'new_model_calls':0,'readiness':'held'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['stage','prepare-supply','register'])
    for k in ('repo','data'): p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','manifest','out','stage-record','prepared'): p.add_argument('--'+k,type=Path)
    p.add_argument('--evaluation-config',type=Path)
    a=p.parse_args()
    if a.mode!='stage' and a.evaluation_config is not None:p.error('later stages use the pinned stage evaluation configuration')
    if a.mode=='stage': result=stage(a.repo,a.data,a.candidate,a.manifest,a.out,read(a.evaluation_config) if a.evaluation_config else None)
    elif a.mode=='prepare-supply': result=prepare_supply(a.repo,a.data,a.stage_record,a.out)
    else: result=register(a.repo,a.data,a.stage_record,a.prepared)
    print(json.dumps(result,sort_keys=True))
