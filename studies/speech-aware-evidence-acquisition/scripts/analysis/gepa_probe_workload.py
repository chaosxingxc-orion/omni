"""Register one fixed16 shared-P GEPA seed mechanism probe through shared Flight.

CPU preparation never admits or dispatches. The seed is the original Solver
text, so this checks native layout/recovery/cost, not learned quality gains.
"""
import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
import tfrl_h1_observation as observation
from tfrl_hardset_a0_registration import pin, pinned, scientific_hash
from tfrl_hardset_a0_workload import read, raw, sha, confined
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

KIND = 'gepa-solver-template-probe'
NAME = 'slurp-gepa-seed-mechanism16-v1'
RUN = 'SAEA-SLURP-GEPA-SEED-MECHANISM16-V1'
ROSTER = 'docs/receipts/rosters/slurp-gepa-seed-mechanism16-v1.json'
AUTH = 'docs/decisions/Agent-Small-Development-Lock.md'
ROWS = 16


def material_checked(repo, data, material_pin):
    """Replay material construction from separately pinned source artifacts."""
    import gepa_seed_material
    if not isinstance(material_pin, dict) or set(material_pin) != {'relpath', 'sha256'}:
        raise ValueError('externally pinned repository seed material required')
    material = pinned(repo, material_pin)
    checked = gepa_seed_material.verify(repo, data, material)
    if material.get('schema') != 'saea-gepa-seed-material-v1':
        raise ValueError('exact seed mechanism material required')
    return material, checked


def population(repo, partition_pin, partition=None):
    from speech_aware_evidence_acquisition.rails.gepa_partition import digest
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    h1, view = observation.population(repo)
    partition = pinned(repo, partition_pin) if partition is None else partition
    if (partition.get('schema') != 'saea-gepa-partition-v1'
            or partition.get('partition_sha256') != digest({k:v for k,v in partition.items()
                if k not in ('partition_sha256', 'input_files_sha256')})):
        raise ValueError('fixed GEPA partition integrity differs')
    expected_inputs = (observation.H1, observation.H2,
        f'configs/dataset/{observation.DATASET}.json', 'configs/dataset/slurp-speech-native-test-full.json')
    pins = partition.get('input_files_sha256', {})
    if set(pins) != set(expected_inputs) or any(sha(confined(repo,p)) != pins[p] for p in expected_inputs):
        raise ValueError('partition original population pins changed')
    feedback, selection = partition['feedback_ids'], partition['selection_ids']
    h1_ids = [r['sample_id'] for r in h1['items']]
    if (len(feedback) != 233 or len(selection) != 80 or len(set(feedback+selection)) != 313
            or set(feedback+selection) != set(h1_ids)
            or feedback != sorted(feedback) or selection != sorted(selection)):
        raise ValueError('exact fixed233/80 H1 partition required')
    ids = feedback[:ROWS]
    by_id = {r['sample_id']: r for r in h1['items']}
    items = [copy.deepcopy(by_id[s]) for s in ids]
    roster = {'schema':'saea-roster-receipt-v1', 'carrier_lock_key':'slurp',
        'roster_id':'SLURP-GEPA-SEED-MECHANISM16-V1', 'roster_role':'construction',
        'split_role':'discovery', 'count':ROWS, 'items':items,
        'split_identity_hash':split_identity_hash(prefixed('slurp', ids)),
        'strata_counts':dict(sorted(Counter(r['stratum'] for r in items).items())),
        'parent':pin(repo,observation.H1), 'partition':partition_pin,
        'selection_rule':'First16 ordered feedback IDs of the existing fixed233/80 partition; no redraw.',
        'claims_benefit':False}
    return roster, view


def derive(repo, data, material_pin):
    material, checked = material_checked(repo, data, material_pin)
    binding = checked['runtime_binding']
    observation.runtime_binding(repo, binding)
    roster, view = population(repo, material['source_pins']['partition'], checked['partition'])
    base = copy.deepcopy(checked['compiled_baseline'])
    source = checked['source_baseline']
    # Material verifier owns exact recompilation; these are independent scientific invariants.
    for key in ('knowledge_stores', 'phase_material', 'thinking_policy', 'phase_tools'):
        if base.get(key) != source.get(key):
            raise ValueError('probe changed frozen ' + key)
    if source.get('thinking_policy') != {role:observation.THINKING_ARMS['baseline'] for role in ('perception','solver')}:
        raise ValueError('seed probe requires original3072 P/S reference')
    base.pop('prepared_phase_supply', None)
    base['native_contact_fit'].update(run_id=RUN+'-FIT', attempt_id='fit-v1', contact_cap=3*ROWS)
    base['what_this_is'] = 'Fixed16 GEPA seed mechanism: one P, gepa Solver, unchanged delivering lock; no learning claim.'
    exp = read(repo/observation.SOURCE_EXP)
    cell, = exp['cells']
    cell.update(cell_id=NAME, baseline_fragment=NAME, model_fragment=binding['model_fragment'],
        run_id=RUN, attempt_id='generation-v1', allowed_execution_stages=['check','prepare-supply','worker','campaign'],
        readiness={'state':'held','requirements':[], 'reason':'Prepare, register, admit and preflight exact GEPA probe before dispatch.'})
    exp.update(experiment_id=RUN, purpose=base['what_this_is'], service_session=binding['service_session'])
    exp['population'].update(count=ROWS, roster_receipt=ROSTER, split_identity_hash=roster['split_identity_hash'])
    exp['populations'] = [{'dataset_fragment':observation.DATASET, 'ids':[r['sample_id'] for r in roster['items']]}]
    exp['execution_plan'].update(planned_model_calls=6*ROWS,
        planned_speech_audio_seconds=math.ceil(6*math.fsum(r['audio_seconds'] for r in roster['items'])))
    exp['stage3_algorithm'] = {'id':'shared-rollout', 'operators_used':['rollout','scoring'],
        'note':'GEPA seed same-P mechanism probe; no prompt mutation, search acceptance or quality claim.'}
    exp['stage3_measurement'].update(rows=ROWS, read_roster=roster['roster_id'], claims_benefit=False,
        expected_effect_pp=0, note=base['what_this_is'])
    return base, exp, roster, view, binding


def fresh_output(data, out):
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('fresh owned CPU output required')


def stage(repo, data, material, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment(); head = require_clean_study_repo(repo)
    observation.require_unspent(repo, data, RUN); fresh_output(data, out)
    paths = [repo/f'configs/{kind}/{NAME}.json' for kind in ('baseline','experiment')] + [repo/ROSTER]
    if any(p.exists() for p in paths):
        raise ValueError('fresh probe configuration and roster required')
    base, exp, roster, _, binding = derive(repo, data, material)
    for path, doc in zip(paths, (base, exp, roster)):
        publish_new(path, raw(doc))
    out.mkdir(parents=True)
    record = {'schema':'saea-gepa-probe-stage-v1', 'cell':NAME, 'run_id':RUN, 'rows':ROWS,
        'material':material, 'runtime_binding':binding, 'source_commit':head,
        'dataset':pin(repo,f'configs/dataset/{observation.DATASET}.json'),
        'source_experiment':pin(repo,observation.SOURCE_EXP),
        'output_hashes':{str(p.relative_to(repo)):sha(p) for p in paths},
        'normal_generation_calls':3*ROWS, 'fallback_inclusive_cap':6*ROWS, 'fit_calls':3*ROWS,
        'new_model_calls':0, 'generation_authorized':False, 'claims_benefit':False}
    publish_new(out/'complete.json', raw(record)); return record


def verify_stage(repo, data, path):
    staged = read(path)
    if (staged.get('schema') != 'saea-gepa-probe-stage-v1' or staged['cell'] != NAME
            or staged['run_id'] != RUN or staged['rows'] != ROWS):
        raise ValueError('exact fixed16 probe stage required')
    for rel, digest in staged['output_hashes'].items():
        if sha(confined(repo, rel)) != digest:
            raise ValueError('staged probe configuration changed')
    pinned(repo, staged['source_experiment']); pinned(repo, staged['dataset'])
    base, exp, roster, _, binding = derive(repo, data, staged['material'])
    expected = {f'configs/baseline/{NAME}.json':base, f'configs/experiment/{NAME}.json':exp, ROSTER:roster}
    if set(staged['output_hashes']) != set(expected) or any(read(repo/p) != doc for p,doc in expected.items()):
        raise ValueError('stage differs from verified material derivation')
    if staged['runtime_binding'] != binding:
        raise ValueError('stage runtime differs')
    return staged


def prepare(repo, data, stage_record, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from tfrl_odd40_cells import verify_output
    assert_execution_environment(); require_clean_study_repo(repo)
    observation.require_unspent(repo, data, RUN); fresh_output(data, out)
    staged = verify_stage(repo, data, stage_record)
    runtime = build_flight_runtime(repo, data, model=staged['runtime_binding']['model_fragment'],
        dataset=observation.DATASET, baseline=NAME, experiment=NAME, run_id=RUN, attempt_id='generation-v1',
        runtime_root=data/'work/scratch/runtime-root', lock_path=repo.parents[1]/'docs/datasets.lock.json')
    target = data/'prepared'/RUN/'phase-supply-v1.json'
    runtime.prepare_phase_supply(target, max_workers=16)
    verify_output(read(target)['output'], [r['sample_id'] for r in read(repo/ROSTER)['items']])
    out.mkdir(parents=True)
    result = {'schema':'saea-gepa-probe-prepared-v1', 'stage_sha256':sha(stage_record), 'rows':ROWS,
        'run_id':RUN, 'supply':pin(data,str(target.relative_to(data))), 'new_model_calls':0}
    publish_new(out/'complete.json', raw(result)); return result


def validate(repo, data, name, spec, exp, view):
    from tfrl_odd40_cells import verify_output
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    base, expected, roster, expected_view, binding = derive(repo, data, spec['material'])
    cell, = exp['cells']
    if spec.get('source_experiment') != pin(repo, observation.SOURCE_EXP):
        raise ValueError('source experiment changed after staging')
    fixed = {'cell':NAME,'experiment':NAME,'baseline':NAME,'run_id':RUN,'attempt_id':'generation-v1',
        'fit_run_id':RUN+'-FIT','fit_attempt_id':'fit-v1','rows':ROWS,'workload_kind':KIND,
        'model_fragment':binding['model_fragment'],'runtime_binding':binding,'profile':binding['profile'],
        'service_session':binding['service_session'],'runtime_receipt':binding['runtime_receipt'],
        'split_identity_hash':roster['split_identity_hash'],'execution':{'admitted_level':96}}
    if name != NAME or any(spec.get(k) != v for k,v in fixed.items()):
        raise ValueError('exact GEPA probe runtime identity required')
    if (view != expected_view or pinned(repo,spec['dataset']) != view or pinned(repo,spec['roster']) != roster
            or spec['roster']['relpath'] != ROSTER):
        raise ValueError('fixed probe population changed')
    for key in ('population','populations','execution_plan','stage3_algorithm','stage3_measurement','service_session'):
        if exp.get(key) != expected.get(key):
            raise ValueError('probe population, budget or declared algorithm changed')
    for key in ('cell_id','baseline_fragment','model_fragment','dataset_fragment','run_id','attempt_id'):
        if cell.get(key) != expected['cells'][0].get(key):
            raise ValueError('registered probe cell changed')
    actual = read(repo/f'configs/baseline/{NAME}.json')
    if (scientific_hash(actual) != scientific_hash(base)
            or scientific_hash(actual) != spec['scientific_baseline_sha256']
            or not native_contact_custody_enabled({**actual,'transport':'vllm','carrier_lock_key':'slurp'})):
        raise ValueError('registered probe program, material or recovery changed')
    if spec['gain_lock'] != {'relpath':observation.LOCK,'sha256':observation.LOCK_SHA} or pinned(repo,spec['gain_lock'])['status'] != 'locked_full_train_fold_b':
        raise ValueError('locked first-round baseline required')
    verify_output(pinned(data,actual['prepared_phase_supply'])['output'], [r['sample_id'] for r in roster['items']])
    return ROWS


def register(repo, data, stage_record, prepared):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment, ExecutionPlan
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
    assert_execution_environment(); head = require_clean_study_repo(repo)
    observation.require_unspent(repo,data,RUN); staged = verify_stage(repo,data,stage_record); prep = read(prepared)
    if (prep.get('schema') != 'saea-gepa-probe-prepared-v1' or prep['stage_sha256'] != sha(stage_record)
            or prep['rows'] != ROWS or prep['run_id'] != RUN or prep['new_model_calls'] != 0):
        raise ValueError('exact prepared probe required')
    bp, ep = [repo/f'configs/{kind}/{NAME}.json' for kind in ('baseline','experiment')]
    base, exp = read(bp), read(ep); pkg = repo/'docs/readiness'/NAME
    if pkg.exists(): raise ValueError('registration exists')
    base['prepared_phase_supply'] = prep['supply']
    exp['cells'][0]['readiness']['requirements'] = [{'kind':'preparation',**prep['supply']}]
    bp.write_bytes(raw(base)); ep.write_bytes(raw(exp))
    binding = staged['runtime_binding']; model = binding['model_fragment']
    cfg = compose(repo,model,observation.DATASET,NAME,NAME); split = exp['population']['split_identity_hash']
    ph = protocol_hash(cfg.config_hash,cfg.values['arm'],split)
    identity = {'cell':NAME,'experiment':NAME,'run_id':RUN,'attempt_id':'generation-v1',
        'fit_run_id':RUN+'-FIT','fit_attempt_id':'fit-v1','rows':ROWS,'composed_config_hash':cfg.config_hash,
        'protocol_hash':ph,'split_identity_hash':split}
    spec = {**identity, 'workload_kind':KIND,'model_fragment':model,'baseline':NAME,'runtime_binding':binding,
        'profile':binding['profile'],'service_session':binding['service_session'],'runtime_receipt':binding['runtime_receipt'],
        'authority':AUTH,'decision_id':NAME+'-owner-admission','decision_relpath':f'coordination/{NAME}/root-admission-v1.json',
        'roster':pin(repo,ROSTER),'dataset':staged['dataset'],'gain_lock':pin(repo,observation.LOCK),
        'material':staged['material'],'source_experiment':staged['source_experiment'],'execution':{'admitted_level':96},
        'scientific_baseline_sha256':scientific_hash(base),'stage_sha256':sha(stage_record),
        'generation_program':'One shared P then seed gepa S and original delivering lock S; no mutation/search.'}
    validate(repo,data,NAME,spec,exp,read(repo/spec['dataset']['relpath']))
    protocol = {'schema':'saea-gepa-probe-generation-v1','cells':{'arm':identity},'profile':binding['profile'],
        'roster':spec['roster'],'material':spec['material'],'normal_generation_calls':48,
        'fallback_inclusive_cap':96,'fit_calls':48,'claims_benefit':False}
    protocol_pin = {'relpath':str((pkg/'protocol-registered-v1.json').relative_to(repo)),
        'sha256':hashlib.sha256(raw(protocol)).hexdigest()}
    docs = {'candidate.json':spec,'protocol-registered-v1.json':protocol}
    rows = []
    for kind,rid,aid,cap in [('generation',RUN,'generation-v1',96),('fit',RUN+'-FIT','fit-v1',48)]:
        budget = {**exp['execution_plan'],'planned_model_calls':cap}
        ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',split_identity_hash=split,protocol_hash=ph,**budget).validate(source_root=repo)
        doc = {'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':NAME,'experiment':NAME,
            'composed_config_hash':cfg.config_hash,'expected_session_id':binding['service_session'],
            'execution_plan':{**budget,'schema':'saea-execution-plan-v1','study':'speech-aware-evidence-acquisition',
                'run_id':rid,'attempt_id':aid,'carrier_lock_key':'slurp','split_identity_hash':split,'protocol_hash':ph},
            'prepared_phase_supply':prep['supply'],'protocol_document':protocol_pin,'status':'held'}
        rel=f'{kind}-plan-registered-v1.json'; docs[rel]=doc
        rows.append(['2026-11-03',rid,'bounded-discovery-probe','GEPA fixed16 seed mechanism '+kind,
            model,'slurp TRAIN feedback fixed16','discovery',split,'no',str(cap),str(budget['planned_speech_audio_seconds']),
            str(budget['planned_gpu_hours']),'All historical costs retained; not a search result',str((pkg/rel).relative_to(repo)),
            'Plan SHA256 '+hashlib.sha256(raw(doc)).hexdigest()+'; held'])
    docs['analysis-recipe-v1.json'] = {'schema':'saea-gepa-probe-read-v1','rows':ROWS,'generation_run_id':RUN,
        'attempt_id':'generation-v1','roster':spec['roster'],'arms':['gepa','lock'],'delivered_arm':'lock',
        'new_model_calls':0,'claims_benefit':False,'candidate_accepted':False}
    pkg.mkdir(parents=True)
    for rel,doc in docs.items(): publish_new(pkg/rel,raw(doc))
    ledger=repo/'docs/exposure-ledger.md'
    ledger.write_text(ledger.read_text()+'\n## GEPA fixed16 seed mechanism\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n')
    publish_new(repo/'docs/readiness'/f'{NAME}-preregistration.md',b'# GEPA seed mechanism probe\n\nFixed first16 feedback IDs; original seed only. 48 normal P/S/S calls,96 fallback cap,48 FIT. Shared P, complete gepa/lock official scoring. Commit, admit and preflight before detached shared dispatch. Preserve all spend and unknown-call barriers; no quality claim.\n')
    return {'cell':NAME,'rows':ROWS,'source_commit':head,'new_model_calls':0,'readiness':'held'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['stage','prepare','register'])
    for key in ('repo','data'): parser.add_argument('--'+key,type=Path,required=True)
    for key in ('material','out','stage-record','prepared'): parser.add_argument('--'+key,type=Path)
    args=parser.parse_args()
    if args.mode=='stage': result=stage(args.repo,args.data,read(args.material),args.out)
    elif args.mode=='prepare': result=prepare(args.repo,args.data,args.stage_record,args.out)
    else: result=register(args.repo,args.data,args.stage_record,args.prepared)
    print(json.dumps(result,sort_keys=True))
