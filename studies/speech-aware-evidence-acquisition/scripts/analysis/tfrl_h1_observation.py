"""Prepare one fixed H1-only fresh P/S observation; no automatic admission.

The observation reuses the original625 dataset and exact313 H1 order. Fresh requests
are not a claim of independent stochastic draws: decoding remains the lock's.
"""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from tfrl_hardset_a0_workload import read, raw, sha, confined
from tfrl_hardset_a0_registration import pin, pinned, scientific_hash
from tfrl_current_generation import MODEL, LOCK, LOCK_SHA, AUTH
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

KIND = 'tfrl-h1-fresh-observation'
H1 = 'docs/receipts/rosters/slurp-stage3-construction-r01c-h1.json'
H2 = 'docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json'
DATASET = 'slurp-speech-native-stage3-construction-r01c'
SOURCE_BASE = 'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json'
SOURCE_EXP = 'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json'
ROUNDS = (1, 2)
THINKING_ARMS = {'off': {'enable_thinking': False, 'answer_reserve_tokens': 1024},
    'short': {'enable_thinking': True, 'reasoning_budget_tokens': 512, 'answer_reserve_tokens': 1024},
    'baseline': {'enable_thinking': True, 'reasoning_budget_tokens': 3072, 'answer_reserve_tokens': 1024}}


def comparison_config(value):
    if (not isinstance(value, dict) or set(value) != {'schema', 'arm', 'admitted_level'}
            or value['schema'] not in ('saea-h1-perception-thinking-v1', 'saea-h2-perception-thinking-v1')
            or value['arm'] not in THINKING_ARMS or type(value['admitted_level']) is not int
            or value['admitted_level'] != 96):
        raise ValueError('exact bounded H1 perception thinking comparison required')
    if value['schema'] == 'saea-h2-perception-thinking-v1' and value['arm'] not in ('off', 'baseline'):
        raise ValueError('H2 screens only the frozen off candidate and baseline')
    return copy.deepcopy(value)


def population_scope(comparison=None):
    half = 'h2' if comparison is not None and comparison_config(comparison)['schema'] == 'saea-h2-perception-thinking-v1' else 'h1'
    return half, H2 if half == 'h2' else H1, 312 if half == 'h2' else 313


def frozen_candidate(repo, comparison):
    if population_scope(comparison)[0] != 'h2':
        return None
    path = 'configs/evaluation/agent-thinking-candidate-freeze-v1.json'
    value = read(repo/path)
    if (value['status'] != 'frozen-for-H2-screening-not-adopted' or value['candidate'] != 'off'
            or value['reference'] != 'baseline' or value['H2_rows'] != 312
            or value['H2_used_to_choose_this_candidate'] is not False
            or value['candidate_thinking_policy'] != {'perception': THINKING_ARMS['off'], 'solver': THINKING_ARMS['baseline']}):
        raise ValueError('exact pre-H2 candidate freeze required')
    return pin(repo, path)


def identity(round_index, comparison=None):
    if comparison is not None:
        cfg = comparison_config(comparison)
        if round_index != 1 or type(round_index) is not int:
            raise ValueError('thinking comparison has one fixed pass per arm')
        arm = cfg['arm']; half = population_scope(cfg)[0]
        return (f'slurp-{half}-perception-thinking-{arm}-v1', f'SAEA-SLURP-{half.upper()}-PERCEPTION-THINKING-{arm.upper()}-V1')
    if type(round_index) is not int or round_index not in ROUNDS:
        raise ValueError('only two bounded fixed-H1 observations are permitted')
    return (f'slurp-tfrl-h1-fresh-observation-r{round_index}-v1',
            f'SAEA-SLURP-TFRL-H1-FRESH-OBSERVATION-R{round_index}-V1')


def predecessor(repo, data, round_index, comparison=None):
    """The second full pass measures conditional repeatability, never iid hardness."""
    identity(round_index, comparison)
    if comparison is not None:
        return None
    if round_index == 1:
        return None
    from speech_aware_evidence_acquisition.core.bundle.verify import verify_bundle
    run = identity(1)[1]
    manifest = data/'runs'/run/'generation-v1'/(run+'.run-manifest.json')
    checked = verify_bundle(manifest)
    plan = checked.manifest['plan']
    roster, _ = population(repo)
    if (checked.attempt_state != 'FINALIZED' or plan['run_id'] != run
            or plan['attempt_id'] != 'generation-v1'
            or plan['split_identity_hash'] != roster['split_identity_hash']):
        raise ValueError('verified full H1 first observation required before second pass')
    return {'manifest': pin(data,str(manifest.relative_to(data))),
            'purpose': 'One full H1 conditional repeatability pass, including correct controls; not iid hardness.',
            'independent_stochastic_draws_claimed': False,
            'maximum_additional_passes': 1}


def population(repo, comparison=None):
    """Read fixed metadata only; no answer/score/test selection channel."""
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    h1, h2 = read(repo/H1), read(repo/H2)
    view = read(repo/f'configs/dataset/{DATASET}.json')
    parent = read(repo/h1['parent']['relpath'])
    if sha(repo/h1['parent']['relpath']) != h1['parent']['file_sha256']:
        raise ValueError('fixed parent roster changed')
    if h1['items'] != parent['items'][::2] or h2['items'] != parent['items'][1::2]:
        raise ValueError('exact fixed interleaved H1/H2 membership required')
    ids = [r['sample_id'] for r in h1['items']]
    blocked = {r['sample_id'] for r in h2['items']}
    uid = lambda sid: ':'.join(sid.split(':')[:3])
    media = {r['sample_id']:r for r in view['items']}
    if (len(ids) != 313 or len(set(ids)) != 313 or len(blocked) != 312
            or len(media) != 625 or len(view['items']) != 625 or view['count'] != 625
            or set(ids) & blocked or {uid(s) for s in ids} & {uid(s) for s in blocked}
            or set(media) != set(ids) | blocked
            or h1['split_identity_hash'] != split_identity_hash(prefixed('slurp', ids))):
        raise ValueError('exact disjoint H1 313 within original625 required')
    hashes = set()
    for r in h1['items']:
        v = media[r['sample_id']]
        if (not r['sample_id'].startswith('slurp:train:') or '/slurp_real/' not in v['media_relpath']
                or not isinstance(v['audio_sha256'], str) or len(v['audio_sha256']) != 64
                or v['audio_seconds'] != r['audio_seconds']
                or not math.isfinite(r['audio_seconds']) or r['audio_seconds'] <= 0):
            raise ValueError('real fixed TRAIN media required')
        hashes.add(v['audio_sha256'])
    if hashes & {media[s]['audio_sha256'] for s in blocked}:
        raise ValueError('H1/H2 audio hash overlap')
    if population_scope(comparison)[0] == 'h2':
        if h2['split_identity_hash'] != split_identity_hash(prefixed('slurp', [r['sample_id'] for r in h2['items']])):
            raise ValueError('H2 split identity changed')
        for row in h2['items']:
            item = media[row['sample_id']]
            if ('/slurp_real/' not in item['media_relpath'] or row['audio_seconds'] != item['audio_seconds']
                    or not math.isfinite(row['audio_seconds']) or row['audio_seconds'] <= 0):
                raise ValueError('fixed real H2 media required')
        return h2, view
    return h1, view


def runtime_binding(repo, binding):
    if set(binding) != {'model_fragment','service_session','profile','runtime_receipt'} or binding['model_fragment'] != MODEL:
        raise ValueError('exact explicit runtime binding required')
    for key in ('model_fragment','service_session','profile'):
        value = binding[key]
        if not isinstance(value,str) or not value or Path(value).name != value or value in ('.','..'):
            raise ValueError('confined runtime identity required')
    model = read(confined(repo, f'configs/model/{MODEL}.json'))
    profile = read(confined(repo, f"configs/deployment/{binding['profile']}.json"))
    if model['runtime_receipt'] != binding['runtime_receipt']:
        raise ValueError('model runtime receipt differs')
    if (profile.get('tensor_parallel_size') != 4 or profile.get('max_num_seqs') != 128
            or profile.get('max_num_batched_tokens') != 2048 or len(profile.get('instances', [])) != 2
            or sorted(c for i in profile['instances'] for c in i['cards']) != list(range(8))):
        raise ValueError('fixed 8-card two-TP4 deployment required')
    if profile['model_fragment'] != MODEL:
        deployed = read(confined(repo, f"configs/model/{profile['model_fragment']}.json"))
        protocol = lambda d:{k:v for k,v in d.items() if k != 'geometry_basis'}
        if protocol(deployed) != protocol(model):raise ValueError('deployment model differs')
    confined(repo/'docs', binding['runtime_receipt'])
    return model


def require_unspent(repo, data, run):
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
    ledger = resolve_asset(repo,'docs/exposure-ledger.md').read_text()
    for rid in (run, run+'-FIT'):
        if '| '+rid+' |' in ledger or any((data/k/rid).exists() for k in ('runs','attempts')):
            raise ValueError('existing registration or spend; never blindly repeat')


def derive(repo, round_index, binding, comparison=None):
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    name, run = identity(round_index, comparison)
    half, roster_path, rows = population_scope(comparison)
    frozen_candidate(repo, comparison)
    roster, view = population(repo, comparison)
    model = runtime_binding(repo, binding)
    base, exp = read(repo/SOURCE_BASE), read(repo/SOURCE_EXP)
    block = base['native_episode']['consumption_module']
    cm.verify_declared_module(block,base,repo_root=repo)
    compiled = cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    if (compiled.program_ceiling != 2 or block['columns'] or block.get('review_loop') != 'off'
            or any(c.get('state') == 'on' and c['component_id'] == 'tfgrpo-solver-experience' for c in block['components'])):
        raise ValueError('unchanged experience-free two-contact lock required')
    # Baseline fragments override model geometry during composition. Bind the
    # legacy lock to the explicitly selected resident context before preparation.
    from speech_aware_evidence_acquisition.rails.adapters.runtime_view import configured_geometry
    argv = model['server_args']
    if argv.count('--max-model-len') != 1:raise ValueError('explicit resident context required')
    base['serving_slot_tokens'] = int(argv[argv.index('--max-model-len')+1])
    configured_geometry({**model, **base}, defaults=(8,8192,64))
    base.pop('prepared_phase_supply',None)
    base['native_contact_fit'].update(run_id=run+'-FIT',attempt_id='fit-v1',contact_cap=2*rows)
    base['what_this_is']='Fixed H1 fresh P/S difficulty observation; no candidate or benefit claim.'
    cell, = exp['cells']
    cell.update(cell_id=name, baseline_fragment=name, model_fragment=MODEL,run_id=run,attempt_id='generation-v1',
                allowed_execution_stages=['check','prepare-supply','worker','campaign'],
                readiness={'state':'held','requirements':[],'reason':'Fixed H1; prepare, register and admit before dispatch.'})
    exp.update(experiment_id=run,purpose=base['what_this_is'],service_session=binding['service_session'])
    exp['population'].update(count=rows,roster_receipt=roster_path,split_identity_hash=roster['split_identity_hash'])
    exp['populations']=[{'dataset_fragment':DATASET,'ids':[r['sample_id'] for r in roster['items']]}]
    exp['execution_plan'].update(planned_model_calls=4*rows,planned_speech_audio_seconds=math.ceil(4*sum(r['audio_seconds'] for r in roster['items'])))
    exp['stage3_algorithm']={'id':'shared-rollout','operators_used':['rollout','scoring'],
                            'note':'Fresh P then S; original lock policy; no H2 score or candidate feedback.'}
    exp['stage3_measurement'].update(rows=rows,read_roster=roster['roster_id'],claims_benefit=False,
        note='One fixed fresh-request observation; no independent stochastic errors or confirmed hardness claim.')
    if comparison is not None:
        cfg = comparison_config(comparison)
        # Only P's policy changes; each independent chain feeds its own P to S.
        base['thinking_policy'] = {'perception': copy.deepcopy(THINKING_ARMS[cfg['arm']]),
                                   'solver': copy.deepcopy(THINKING_ARMS['baseline'])}
        base['what_this_is'] = 'Fixed '+half.upper()+' Agent comparison: P thinking '+cfg['arm']+'; same S, knowledge, prompts and real audio.'
        exp['purpose'] = base['what_this_is']
        exp['stage3_measurement']['note'] = base['what_this_is']+' TRAIN development comparison, not test gain.'
        exp['stage3_algorithm']['note'] = 'P-only thinking policy ablation; no learned experience or H2 feedback.'
        if half == 'h2':
            exp['stage3_algorithm']['note'] = 'Frozen H1-selected policy screening on H2; no label feedback to learning.'
    return base, exp, roster, view


def stage(repo, data, round_index, binding, out, comparison=None):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    name,run=identity(round_index,comparison);require_unspent(repo,data,run)
    previous=predecessor(repo,data,round_index,comparison)
    paths=[repo/f'configs/{k}/{name}.json' for k in ('baseline','experiment')]
    if out.exists() or any(p.exists() for p in paths) or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned stage and configuration required')
    base,exp,roster,view=derive(repo,round_index,binding,comparison=comparison) if comparison is not None else derive(repo,round_index,binding)
    half, roster_path, rows = population_scope(comparison)
    out.mkdir(parents=True)
    for path,doc in zip(paths,(base,exp)):publish_new(path,raw(doc))
    result={'schema':'saea-h1-observation-stage-v1','round':round_index,'cell':name,'run_id':run,'rows':rows,
        'runtime_binding':binding,'source_commit':head,'dataset':pin(repo,f'configs/dataset/{DATASET}.json'),
        'h1':pin(repo,H1),'h2_metadata':pin(repo,H2),'source_baseline':pin(repo,SOURCE_BASE),
        'output_hashes':{str(p.relative_to(repo)):sha(p) for p in paths},'normal_generation_calls':2*rows,
        'fallback_inclusive_cap':4*rows,'fit_calls':2*rows,'new_model_calls':0,'test_labels_read':False,
        'h2_scores_read':False,'prior_costs_reused_or_reset':False,'generation_authorized':False}
    if previous is not None:result['predecessor']=previous
    if comparison is not None:result['comparison']=comparison_config(comparison)
    if half == 'h2':result['candidate_freeze'] = frozen_candidate(repo, comparison)
    publish_new(out/'complete.json',raw(result));return result


def prepare(repo,data,stage_record,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from tfrl_odd40_cells import verify_output
    assert_execution_environment();require_clean_study_repo(repo)
    staged=read(stage_record);comparison=staged.get('comparison')
    half, roster_path, rows = population_scope(comparison)
    if half == 'h2' and staged.get('candidate_freeze') != frozen_candidate(repo, comparison):
        raise ValueError('H2 candidate freeze changed')
    name,run=identity(staged['round'],comparison);require_unspent(repo,data,run)
    if staged.get('predecessor')!=predecessor(repo,data,staged['round'],comparison):raise ValueError('observation predecessor changed')
    for rel,digest in staged['output_hashes'].items():
        if sha(confined(repo,rel))!=digest:raise ValueError('stage changed')
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned preparation required')
    runtime=build_flight_runtime(repo,data,model=MODEL,dataset=DATASET,baseline=name,experiment=name,
        run_id=run,attempt_id='generation-v1',runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json')
    target=data/'prepared'/run/'phase-supply-v1.json';runtime.prepare_phase_supply(target,max_workers=16)
    ids=[r['sample_id'] for r in population(repo, comparison)[0]['items']];verify_output(read(target)['output'],ids)
    out.mkdir(parents=True);result={'schema':'saea-h1-observation-prepared-v1','stage_sha256':sha(stage_record),
        'rows':rows,'run_id':run,'supply':pin(data,str(target.relative_to(data))),'new_model_calls':0}
    publish_new(out/'complete.json',raw(result));return result


def validate(repo,data,name,spec,exp,view):
    from tfrl_odd40_cells import verify_output
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    comparison=spec.get('comparison')
    half, roster_path, rows = population_scope(comparison)
    if half == 'h2' and spec.get('candidate_freeze') != frozen_candidate(repo, comparison):
        raise ValueError('registered H2 candidate freeze changed')
    expected_name,run=identity(spec['round'],comparison);binding=spec['runtime_binding']
    if name!=expected_name or spec['run_id']!=run:raise ValueError('observation round identity differs')
    if spec.get('predecessor')!=predecessor(repo,data,spec['round'],comparison):raise ValueError('observation predecessor changed')
    expected_base,expected_exp,roster,expected_view=derive(repo,spec['round'],binding,comparison)
    cell,=exp['cells'];ids=[r['sample_id'] for r in roster['items']]
    if (name!=expected_name or spec['run_id']!=run or spec['attempt_id']!='generation-v1'
        or spec['fit_run_id']!=run+'-FIT' or spec['fit_attempt_id']!='fit-v1' or spec['rows']!=rows
        or cell['run_id']!=run or cell['attempt_id']!='generation-v1' or cell['baseline_fragment']!=name
        or cell['model_fragment']!=MODEL or spec['model_fragment']!=MODEL
        or spec['service_session']!=binding['service_session'] or exp['service_session']!=binding['service_session']
        or spec['profile']!=binding['profile'] or spec['runtime_receipt']!=binding['runtime_receipt']
        or view!=expected_view or pinned(repo,spec['dataset'])!=view or pinned(repo,spec['roster'])!=roster
        or exp['populations']!=expected_exp['populations'] or exp['population']!=expected_exp['population']
        or spec['split_identity_hash']!=roster['split_identity_hash']
        or exp['execution_plan']!=expected_exp['execution_plan'] or exp['stage3_algorithm']!=expected_exp['stage3_algorithm']
        or exp['stage3_measurement']!=expected_exp['stage3_measurement']):
        raise ValueError('fixed H1 observation identity, population or budget changed')
    base=read(repo/f'configs/baseline/{name}.json')
    if (scientific_hash(base)!=scientific_hash(expected_base) or scientific_hash(base)!=spec['scientific_baseline_sha256']
        or not native_contact_custody_enabled({**base,'carrier_lock_key':'slurp','transport':'vllm'})):
        raise ValueError('fixed lock program or recovery changed')
    if spec['gain_lock']!={'relpath':LOCK,'sha256':LOCK_SHA} or pinned(repo,spec['gain_lock'])['status']!='locked_full_train_fold_b':
        raise ValueError('locked first-round baseline required')
    verify_output(pinned(data,base['prepared_phase_supply'])['output'],ids)
    return rows


def register(repo,data,stage_record,prepared):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment,ExecutionPlan
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment();head=require_clean_study_repo(repo)
    staged,prep=read(stage_record),read(prepared);comparison=staged.get('comparison')
    half, roster_path, rows = population_scope(comparison)
    if half == 'h2' and staged.get('candidate_freeze') != frozen_candidate(repo, comparison):
        raise ValueError('H2 candidate freeze changed')
    name,run=identity(staged['round'],comparison);require_unspent(repo,data,run)
    previous=predecessor(repo,data,staged['round'],comparison)
    if staged.get('predecessor')!=previous:raise ValueError('observation predecessor changed')
    if (staged['schema']!='saea-h1-observation-stage-v1' or prep['schema']!='saea-h1-observation-prepared-v1'
        or prep['stage_sha256']!=sha(stage_record) or prep['rows']!=rows or prep['run_id']!=run):raise ValueError('exact prepared stage required')
    for key in ('dataset','h1','h2_metadata','source_baseline'):pinned(repo,staged[key])
    for rel,digest in staged['output_hashes'].items():
        if sha(confined(repo,rel))!=digest:raise ValueError('staged configuration changed')
    bp=repo/f'configs/baseline/{name}.json';ep=repo/f'configs/experiment/{name}.json';base,exp=read(bp),read(ep)
    pkg=repo/'docs/readiness'/name
    if pkg.exists():raise ValueError('registration exists')
    binding=staged['runtime_binding'];base['prepared_phase_supply']=prep['supply']
    exp['cells'][0]['readiness']['requirements']=[{'kind':'preparation',**prep['supply']}]
    bp.write_bytes(raw(base));ep.write_bytes(raw(exp))
    cfg=compose(repo,MODEL,DATASET,name,name);roster=population(repo, comparison)[0];split=roster['split_identity_hash'];ph=protocol_hash(cfg.config_hash,cfg.values['arm'],split)
    ident={'cell':name,'experiment':name,'run_id':run,'attempt_id':'generation-v1','fit_run_id':run+'-FIT','fit_attempt_id':'fit-v1',
           'rows':rows,'composed_config_hash':cfg.config_hash,'protocol_hash':ph,'split_identity_hash':split}
    spec={**ident,'workload_kind':KIND,'round':staged['round'],'model_fragment':MODEL,'baseline':name,'runtime_binding':binding,
          'service_session':binding['service_session'],'profile':binding['profile'],'runtime_receipt':binding['runtime_receipt'],
          'authority':('docs/decisions/Agent-Small-Development-Lock.md' if comparison is not None else AUTH),'decision_id':name+'-owner-admission','decision_relpath':f'coordination/{name}/root-admission-v1.json',
          'roster':pin(repo,roster_path),'dataset':staged['dataset'],'gain_lock':pin(repo,LOCK),'scientific_baseline_sha256':scientific_hash(base),
          'stage_sha256':sha(stage_record),'generation_program':'Fresh P then S; exact fixed H1, no learned experience.'}
    if previous is not None:spec['predecessor']=previous
    if comparison is not None:
        spec['comparison']=comparison_config(comparison)
        spec['generation_program']='P thinking '+comparison['arm']+' then unchanged thinking S; fixed '+half.upper()+', no learned experience.'
    if half == 'h2':spec['candidate_freeze'] = frozen_candidate(repo, comparison)
    validate(repo,data,name,spec,exp,read(repo/f'configs/dataset/{DATASET}.json'))
    protocol={'schema':'saea-h1-observation-generation-v1','cells':{'arm':ident},'profile':binding['profile'],
              'roster':spec['roster'],'normal_generation_calls':2*rows,'fallback_inclusive_cap':4*rows,'round':staged['round'],
              'total_scheduled_rounds':1,'claims_benefit':False,'independent_stochastic_draws_claimed':False}
    if comparison is not None:protocol['comparison']=comparison_config(comparison)
    docs={'candidate.json':spec,'protocol-registered-v1.json':protocol}
    pp={'relpath':str((pkg/'protocol-registered-v1.json').relative_to(repo)),'sha256':hashlib.sha256(raw(protocol)).hexdigest()}
    ledger_rows=[]
    for kind,rid,aid,cap in [('generation',run,'generation-v1',4*rows),('fit',run+'-FIT','fit-v1',2*rows)]:
        budget={**exp['execution_plan'],'planned_model_calls':cap}
        ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',split_identity_hash=split,protocol_hash=ph,**budget).validate(source_root=repo)
        doc={'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':name,'experiment':name,'composed_config_hash':cfg.config_hash,
             'expected_session_id':binding['service_session'],'execution_plan':{**budget,'schema':'saea-execution-plan-v1','study':'speech-aware-evidence-acquisition',
             'run_id':rid,'attempt_id':aid,'carrier_lock_key':'slurp','split_identity_hash':split,'protocol_hash':ph},
             'prepared_phase_supply':prep['supply'],'protocol_document':pp,'status':'Registered; held pending shared repin/admission/preflight.'}
        rel=f'{kind}-plan-registered-v1.json';docs[rel]=doc
        ledger_rows.append(['2026-11-01',rid,'bounded-discovery-probe','Fixed '+half.upper()+' observation '+kind,MODEL,'slurp TRAIN '+half.upper()+' fixed'+str(rows),'discovery',split,'no',str(cap),str(budget['planned_speech_audio_seconds']),str(budget['planned_gpu_hours']),'Prior observations and all costs retained; no H2 score selection',str((pkg/rel).relative_to(repo)),f'Plan SHA256 {hashlib.sha256(raw(doc)).hexdigest()}; held under {AUTH}'])
    docs['analysis-recipe-v1.json']={'schema':'saea-h1-observation-read-v1','round':staged['round'],'rows':rows,'generation_run_id':run,
        'attempt_id':'generation-v1','roster':spec['roster'],'new_model_calls':0,'claims_benefit':False,
        'scope':f'Officially score complete{rows}; retain all outcomes and costs. No label feedback to learning or automatic adoption.'}
    pkg.mkdir(parents=True)
    for rel,doc in docs.items():publish_new(pkg/rel,raw(doc))
    ledger=repo/'docs/exposure-ledger.md';ledger.write_text(ledger.read_text()+'\n## Fixed H1 fresh observation\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in ledger_rows)+'\n')
    publish_new(repo/'docs/readiness'/f'{name}-preregistration.md',f'# Fixed {half.upper()} observation\n\nExact {rows} rows from original625. Fresh P/S, {2*rows} normal calls, {4*rows} fallback cap, {2*rows} FIT. Preserve historical costs; commit registration, shared admission and preflight before detached dispatch. No automatic adoption.\n'.encode())
    return {'cell':name,'round':staged['round'],'rows':rows,'source_commit':head,'new_model_calls':0,'readiness':'held'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare','register'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('binding','out','stage-record','prepared'):p.add_argument('--'+k,type=Path)
    p.add_argument('--round',dest='round_index',type=int,choices=ROUNDS)
    p.add_argument('--comparison',type=Path,help='Fixed P thinking comparison JSON; stage only.')
    a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='stage':
        a.pop('stage_record');a.pop('prepared');a['binding']=read(a['binding'])
        a['comparison']=read(a['comparison']) if a['comparison'] is not None else None
        result=stage(**a)
    elif mode=='prepare':
        for k in ('binding','round_index','prepared','comparison'):a.pop(k)
        result=prepare(**a)
    else:
        for k in ('binding','round_index','out','comparison'):a.pop(k)
        result=register(**a)
    print(__import__('json').dumps(result,sort_keys=True))
