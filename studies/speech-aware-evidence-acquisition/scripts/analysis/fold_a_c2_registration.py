"""Register and validate full Fold A c2 generation, FIT and official final read.

No model contacts or readiness promotion. A0 remains an evaluation control;
original exposure history and failed candidate stop decisions remain unchanged.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/flights/nx')]
import fold_a_c2_baseline as baseline
import fold_a_c2_resident as resident
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read,sha,raw=baseline.read,baseline.sha,baseline.raw
NAME,RUN=baseline.CELL,baseline.RUN
PROFILE='qwen3-omni-real500-2xtp4-s128-b2048-v1'
AUTH='docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'
CLOSURE='docs/readiness/2026-10-31-fold-a-c2-preparation-closure-v1/closure-v1.json'
LOCK='docs/results/2026-10-31-a0-full1080-baseline-lock.json'
LOCK_SHA='6b76ec8fcc232c2e1b33a9c3221bed93a437f180ab8102ad9706082f5b81ae69'


def scientific_baseline_hash(base):
    doc=copy.deepcopy(base);doc.pop('prepared_phase_supply')
    return hashlib.sha256(raw(doc)).hexdigest()


def pinned(root,pin):
    p=baseline.comparator.confined(root,pin['relpath'])
    if sha(p)!=pin['sha256']:raise ValueError('pinned input changed: '+pin['relpath'])
    return read(p)


def validate(repo,data,name,spec,exp,view):
    cell,=exp['cells'];pop,=exp['populations'];summary=exp['population']
    if (name!=NAME or spec['cell']!=NAME or spec['experiment']!=NAME or spec['run_id']!=RUN
            or spec['attempt_id']!='generation-v1' or spec['rows']!=1080
            or cell['cell_id']!=NAME or cell['run_id']!=RUN or cell['attempt_id']!='generation-v1'
            or cell['baseline_fragment']!=NAME or cell['model_fragment']!=resident.MODEL
            or exp['service_session']!=resident.SESSION or spec['profile']!=PROFILE
            or spec['service_session']!=resident.SESSION or spec['runtime_receipt']!=resident.RECEIPT):
        raise ValueError('exact Fold A c2 identity required')
    roster=pinned(repo,spec['roster']);dataset=pinned(repo,spec['dataset'])
    ids=[r['sample_id'] for r in roster['items']]
    if (len(ids)!=1080 or len(set(ids))!=1080 or dataset!=view or view['count']!=1080
            or len(view['items'])!=1080 or {r['sample_id'] for r in view['items']}!=set(ids)
            or pop['ids']!=ids or summary['count']!=1080 or summary['roster_receipt']!=spec['roster']['relpath']
            or len({roster['split_identity_hash'],view['split_identity_hash'],summary['split_identity_hash'],spec['split_identity_hash']})!=1
            or pop['dataset_fragment']!=cell['dataset_fragment'] or summary['dataset_fragment']!=cell['dataset_fragment']):
        raise ValueError('complete real Fold A roster differs')
    for row in view['items']:
        if (not row['sample_id'].startswith('slurp:train:') or '/slurp_real/' not in row['media_relpath']
                or len(row['audio_sha256'])!=64 or set(row['audio_sha256'])-set('0123456789abcdef')):
            raise ValueError('real training audio identity required')
    base=read(repo/f'configs/baseline/{NAME}.json');fit=base['native_contact_fit']
    if (base['policy']!='phase-native-module-v1-c2' or scientific_baseline_hash(base)!=spec['scientific_baseline_sha256']
            or exp['stage3_algorithm']['id']!='none' or exp['stage3_algorithm']['operators_used']!=[]
            or exp['stage3_measurement']['claims_benefit'] is not False
            or fit['contact_cap']!=2160 or fit['run_id']!=RUN+'-FIT' or fit['attempt_id']!='fit-v1'
            or spec['fit_run_id']!=fit['run_id'] or spec['fit_attempt_id']!=fit['attempt_id']
            or exp['execution_plan']['planned_model_calls']!=4320
            or exp['execution_plan']['planned_speech_audio_seconds']<math.ceil(4*sum(r['audio_seconds'] for r in view['items']))):
        raise ValueError('frozen two-contact method or complete budget differs')
    module=base['native_episode']['consumption_module']
    baseline.cm.verify_declared_module(module,base,repo_root=repo)
    if baseline.cm.compile_module(module,baseline.carrier_module_hooks(repo_root=repo),repo_root=repo).program_ceiling!=2:
        raise ValueError('two-contact program required')
    model=pinned(repo,spec['model'])
    if model!=read(repo/f'configs/model/{resident.MODEL}.json') or model['runtime_receipt']!=resident.RECEIPT:
        raise ValueError('selected decoding changed')
    closure=pinned(repo,spec['preparation_closure'])
    if closure['rows']!=1080 or closure['generation_admitted'] or not closure['derivatives_values_unchanged']:
        raise ValueError('completed preparation required')
    for rel,digest in closure['pins'].items():
        if sha(baseline.comparator.confined(data,rel))!=digest:raise ValueError('prepared input changed')
    ledger=(repo/'docs/exposure-ledger.md').read_text()
    for row in [closure['encoder_ledger_row'],*closure['derivative_ledger_rows']]:
        if ledger.splitlines().count('| '+' | '.join(row)+' |')!=1:raise ValueError('closed preparation ledger missing')
    if closure['closed_agreement_line'] not in ledger:raise ValueError('agreement closure missing')
    supply=pinned(data,base['prepared_phase_supply'])
    from fold_a_c2_supply import verify_output
    verify_output(supply['output'],ids)
    if (spec['gain_lock']!={'relpath':LOCK,'sha256':LOCK_SHA}
            or pinned(repo,spec['gain_lock'])['status']!='locked_full_train_fold_b'):
        raise ValueError('full first-system gain prerequisite changed')
    return 1080


def register(repo,data,stage,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment,ExecutionPlan
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment();head=require_clean_study_repo(repo)
    stage_doc=read(stage)
    if (stage_doc['schema']!='saea-fold-a-c2-resident-stage-v1' or stage_doc['rows']!=1080
            or stage_doc['new_model_calls']!=0 or stage_doc['readiness']!='held' or stage_doc['registered']):
        raise ValueError('complete held runtime staging required')
    for rel,digest in stage_doc['output_hashes'].items():
        if sha(repo/rel)!=digest:raise ValueError('staged configuration changed')
    if sha(repo/CLOSURE)!=stage_doc['closure_sha256']:raise ValueError('staged closure changed')
    pkg=repo/'docs/readiness'/NAME;ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    if pkg.exists() or out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh registration outputs required')
    for rid in (RUN,RUN+'-FIT',RUN+'-READ'):
        if '| '+rid+' |' in text or any((data/k/rid).exists() for k in ('runs','attempts')):
            raise ValueError('existing registration/spend; inspect before continuation')
    exp=read(repo/f'configs/experiment/{NAME}.json');cell,=exp['cells'];base=read(repo/f'configs/baseline/{NAME}.json')
    if cell['readiness']['state']!='held':raise ValueError('held control required')
    def pin(rel):return {'relpath':rel,'sha256':sha(repo/rel)}
    split=exp['population']['split_identity_hash']
    cfg=compose(repo,resident.MODEL,cell['dataset_fragment'],NAME,NAME);ph=protocol_hash(cfg.config_hash,cfg.values['arm'],split)
    identity={'cell':NAME,'experiment':NAME,'run_id':RUN,'attempt_id':'generation-v1','fit_run_id':RUN+'-FIT',
        'fit_attempt_id':'fit-v1','rows':1080,'composed_config_hash':cfg.config_hash,'protocol_hash':ph,'split_identity_hash':split}
    spec={**identity,'workload_kind':'a0-fold-a-c2','model_fragment':resident.MODEL,'baseline':NAME,
        'service_session':resident.SESSION,'profile':PROFILE,'runtime_receipt':resident.RECEIPT,'authority':AUTH,
        'decision_id':NAME+'-owner-admission','decision_relpath':f'coordination/{NAME}/root-admission-v1.json',
        'gain_lock':pin(LOCK),'roster':pin(exp['population']['roster_receipt']),
        'dataset':pin(f"configs/dataset/{cell['dataset_fragment']}.json"),'model':pin(f'configs/model/{resident.MODEL}.json'),
        'preparation_closure':pin(CLOSURE),'scientific_baseline_sha256':scientific_baseline_hash(base),
        'generation_program':'Fresh real-audio Perception then Solver; complete Fold A control, no RL operator.',
        'runtime_stage_sha256':sha(stage)}
    validate(repo,data,NAME,spec,exp,read(repo/spec['dataset']['relpath']))
    protocol={'schema':'saea-fold-a-c2-generation-v1','cells':{'arm':identity},'profile':PROFILE,
        'roster':spec['roster'],'gain_lock':spec['gain_lock'],'preparation_closure':spec['preparation_closure'],
        'prepared_phase_supply':base['prepared_phase_supply'],'normal_generation_calls':2160,'fallback_inclusive_cap':4320,
        'claims_benefit':False,'budget_basis':'16 allocated card-hours is a cap, not a throughput forecast.',
        'exposure':'Historical Fold A exposure retained; no test tuning; candidate evaluation remains separately frozen.'}
    pending={'candidate.json':spec,'protocol-registered-v1.json':protocol}
    protocol_pin={'relpath':str((pkg/'protocol-registered-v1.json').relative_to(repo)),
        'sha256':hashlib.sha256(raw(protocol)).hexdigest()}
    rows=[]
    for kind,rid,aid,calls in [('generation',RUN,'generation-v1',4320),('fit',RUN+'-FIT','fit-v1',2160)]:
        budget={**exp['execution_plan'],'planned_model_calls':calls}
        ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',split_identity_hash=split,protocol_hash=ph,**budget).validate(source_root=repo)
        plan={'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':NAME,'experiment':NAME,
            'composed_config_hash':cfg.config_hash,'expected_session_id':resident.SESSION,
            'execution_plan':{**budget,'schema':'saea-execution-plan-v1','study':'speech-aware-evidence-acquisition',
                'run_id':rid,'attempt_id':aid,'carrier_lock_key':'slurp','protocol_hash':ph,'split_identity_hash':split},
            'prepared_phase_supply':base['prepared_phase_supply'],'protocol_document':protocol_pin,
            'status':'REGISTERED BEFORE CONTACT; held pending shared repin, resident verification, admission and preflight'}
        rel=f'{kind}-plan-registered-v1.json';pending[rel]=plan
        rows.append(['2026-10-31',rid,'bounded-discovery-probe','Full1080 Fold A c2 '+kind,resident.MODEL,
            'slurp TRAIN Fold A','discovery',split,'no',str(calls),str(budget['planned_speech_audio_seconds']),
            str(budget['planned_gpu_hours']),'Historical Fold A exposure and all source costs retained',
            str((pkg/rel).relative_to(repo)),f'Plan SHA256 {hashlib.sha256(raw(plan)).hexdigest()}; root countersigned under {AUTH}; coordinator Codex; held; no test tuning'])
    recipe={'schema':'saea-fold-a-c2-read-v1','read_run_id':RUN+'-READ','generation_run_id':RUN,'attempt_id':'generation-v1',
        'rows':1080,'roster':spec['roster'],'split_identity_hash':split,'score_surface':'sealed-final-delivery',
        'scope':'One complete official SLURP control score from the sealed final answers; no subset selection or tuning.',
        'official_scoring_jobs':1,'new_model_calls':0,'claims_benefit':False,'authority':AUTH,
        'execution':'Shared campaign scoring and finalization after generation lease release; checkpoint custody retained.'}
    pending['analysis-recipe-v1.json']=recipe
    rows.append(['2026-10-31',RUN+'-READ','model-free-check','Complete Fold A A0 official final read','official SLURP scorer',
        'slurp TRAIN Fold A','discovery',split,'no','0','0','0','All generation and FIT costs retained',
        str((pkg/'analysis-recipe-v1.json').relative_to(repo)),f'Recipe SHA256 {hashlib.sha256(raw(recipe)).hexdigest()}; root countersigned under {AUTH}; no candidate gain claim'])
    out.mkdir(parents=True);publish_new(out/'intent.json',raw({'source_commit':head,'runtime_stage_sha256':sha(stage),'new_model_calls':0}))
    pkg.mkdir(parents=True)
    for rel,doc in pending.items():publish_new(pkg/rel,raw(doc))
    ledger.write_text(text+'\n## Full Fold A c2 control\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n')
    note=repo/'docs/readiness'/f'{NAME}-preregistration.md'
    publish_new(note,b'# Full Fold A c2 control\n\n1080 real rows, fresh Perception and Solver. Normal2160, fallback cap4320, separate FIT2160. Full official final delivery read; no tuning or candidate gain claim. Shared repin, observed resident, admission and preflight precede detached dispatch. Preserve checkpoint and unresolved request identities.\n')
    result={'schema':'saea-fold-a-c2-registration-v1','rows':1080,'source_commit':head,'new_model_calls':0,'readiness':'held',
        'normal_generation_calls':2160,'fallback_inclusive_cap':4320,'fit_cap':2160,
        'files':{str((pkg/rel).relative_to(repo)):sha(pkg/rel) for rel in pending}}
    publish_new(out/'complete.json',raw(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','stage','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(register(**vars(p.parse_args()))))
