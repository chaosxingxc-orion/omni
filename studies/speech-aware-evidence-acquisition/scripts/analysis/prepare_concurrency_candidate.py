"""Stage a fresh real100 flight and prepare its first-P supply, without inference."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from concurrency_sweep_workload import validate_manifest, build_execution_view
from registered_workload import validate_workload


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(doc, stream, indent=2, sort_keys=True)
        stream.write('\n')


def configuration(source_experiment, source_baseline, view, *, level, model, session):
    """Preserve the two-contact task, replace every population/spend identity."""
    if type(level) is not int or level not in (32, 48, 64, 96):
        raise ValueError('unsupported client concurrency')
    name = f'slurp-concurrency-c{level}-real100-v1'
    run = f'SAEA-SLURP-CONCURRENCY-C{level}-REAL100-V1'
    dataset = 'slurp-concurrency-real100-v1'
    base = copy.deepcopy(source_baseline)
    base.pop('prepared_phase_supply', None)
    base['serving_slot_tokens'] = 49152
    base['native_contact_fit'].update(run_id=run+'-FIT', attempt_id='fit-v1', contact_cap=200)
    exp = copy.deepcopy(source_experiment)
    cell, = exp['cells']
    for key in ('resume_from', 'checkpoint_compatibility', 'native_contact_recovery'):
        cell.pop(key, None)
    cell.update(cell_id=name, run_id=run, attempt_id='generation-v1',
                dataset_fragment=dataset, baseline_fragment=name, model_fragment=model,
                readiness={'state': 'held', 'reason': 'Fresh real100 preparation; admission pending', 'requirements': []})
    exp.update(experiment_id=run, date='2026-11-02', service_session=session,
               split_id=view['split_name'], purpose='Fixed real100 client-concurrency comparison; fresh P feeds S',
               population={'count': 100, 'dataset_fragment': dataset, 'split_identity_hash': view['split_identity_hash']},
               populations=[{'dataset_fragment': dataset, 'ids': [r['sample_id'] for r in view['items']]}])
    exp['execution_plan'].update(planned_model_calls=400, planned_gpu_hours=4.0,
                                 planned_speech_audio_seconds=math.ceil(4*view['audio_seconds_total']))
    exp['stage3_measurement'].update(rows=100, read_roster=view['split_name'])
    return name, run, dataset, base, exp


def stage(repo, data, source, manifest, roster, parent_view, level, profile_id, session, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.deployment import load_deployment_profile
    assert_execution_environment()
    profile = load_deployment_profile(profile_id, repo)
    if (profile.tensor_parallel_size != 4 or profile.max_num_seqs != 128
            or len(profile.instances) != 2 or profile.admitted_level != level):
        raise ValueError('requires two TP4 instances at declared client concurrency')
    checked = validate_manifest(manifest, source_roster_path=roster,
                                expected_profile_id=profile_id, expected_profile_sha256=profile.sha256)
    view = build_execution_view(checked, read(parent_view))
    # This model ID is the one registered by the current serving coordinator.
    model_id = 'bf16-thinking-real500-tp4-s128-v1'
    model = read(repo/f'configs/model/{model_id}.json')
    receipt = read(repo/'docs'/model['runtime_receipt'])
    if (model['server_args'] != receipt['protocol']['server_args']
            or model['server_args'][model['server_args'].index('--max-model-len')+1] != '49152'
            or model['decoding_params']['max_tokens'] != 4096
            or model['reasoning_budget_tokens'] != 3072):
        raise ValueError('model and resident protocol must match the frozen 48k task budget')
    old = 'slurp-real500-tp4-s128-v1'
    name, run, dataset, base, exp = configuration(
        read(source/f'configs/experiment/{old}.json'), read(source/f'configs/baseline/{old}.json'), view,
        level=level, model=model_id, session=session)
    for rid in (run, run+'-FIT'):
        if any((data/key/rid).exists() for key in ('runs', 'attempts')):
            raise ValueError('spend identity already exists')
    spec = {'cell': name, 'experiment': name, 'run_id': run, 'attempt_id': 'generation-v1',
            'fit_run_id': run+'-FIT', 'fit_attempt_id': 'fit-v1', 'workload_kind': 'concurrency-sweep-real100',
            'rows': 100, 'new_model_calls': 0, 'manifest_path': str(manifest), 'source_roster_path': str(roster),
            'manifest_sha256': checked['manifest_sha256'], 'profile': profile_id, 'profile_sha256': profile.sha256,
            'split_identity_hash': view['split_identity_hash'], 'service_session': session,
            'model_fragment': model_id, 'baseline': name, 'runtime_receipt': model['runtime_receipt'],
            'runtime_receipt_sha256': sha(repo/'docs'/model['runtime_receipt'])}
    outputs = {f'configs/baseline/{name}.json': base, f'configs/experiment/{name}.json': exp,
               f'configs/dataset/{dataset}.json': view}
    if out.exists() or any((repo/p).exists() for p in outputs):
        raise ValueError('fresh output and candidate required')
    for rel, doc in outputs.items():
        write(repo/rel, doc)
    validate_workload(repo, data, name, spec)
    write(out/'stage.json', {'spec': spec, 'output_hashes': {p: sha(repo/p) for p in outputs},
                             'source_experiment_sha256': sha(source/f'configs/experiment/{old}.json'),
                             'source_baseline_sha256': sha(source/f'configs/baseline/{old}.json'),
                             'new_model_calls': 0, 'generation_authorized': False})
    return {'candidate': name, 'rows': 100, 'stage': 'held', 'new_model_calls': 0}


def prepare(repo, data, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment()
    require_clean_study_repo(repo)
    staged = read(out/'stage.json'); spec = staged['spec']; name = spec['cell']
    for rel, digest in staged['output_hashes'].items():
        if sha(repo/rel) != digest:
            raise ValueError('staged configuration changed: '+rel)
    validate_workload(repo, data, name, spec)
    cell, = read(repo/f'configs/experiment/{name}.json')['cells']
    runtime = build_flight_runtime(repo, data, model=cell['model_fragment'], dataset=cell['dataset_fragment'],
                                   baseline=name, experiment=name, run_id=spec['run_id'], attempt_id=spec['attempt_id'],
                                   runtime_root=data/'work/scratch/runtime-root', lock_path=repo.parents[1]/'docs/datasets.lock.json')
    target = data/'prepared'/spec['run_id']/'phase-supply-v1.json'
    runtime.prepare_phase_supply(target, max_workers=16)
    report = {'rows': 100, 'supply': {'relpath': str(target.relative_to(data)), 'sha256': sha(target)},
              'stage_sha256': sha(out/'stage.json'), 'new_model_calls': 0, 'generation_authorized': False}
    write(out/'prepared.json', report)
    return report


def register(repo, data, out):
    """Bind the prepared supply to ordinary campaign plans and exposure records."""
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment, ExecutionPlan
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
    assert_execution_environment(); require_clean_study_repo(repo)
    staged = read(out/'stage.json'); prep = read(out/'prepared.json'); spec = dict(staged['spec'])
    if (prep['stage_sha256'] != sha(out/'stage.json') or prep['rows'] != 100
            or prep['new_model_calls'] != 0 or prep['generation_authorized'] is not False
            or sha(data/prep['supply']['relpath']) != prep['supply']['sha256']):
        raise ValueError('prepared supply binding differs')
    for rel, digest in staged['output_hashes'].items():
        if sha(repo/rel) != digest: raise ValueError('staged configuration changed')
    name = spec['cell']; run = spec['run_id']; pkg = repo/'docs/readiness'/name
    bp = repo/f'configs/baseline/{name}.json'; ep = repo/f'configs/experiment/{name}.json'
    base = read(bp); exp = read(ep); cell, = exp['cells']
    ledger = repo/'docs/exposure-ledger.md'
    if pkg.exists() or any('| '+rid+' |' in ledger.read_text() for rid in (run,run+'-FIT')):
        raise ValueError('candidate already registered')
    validate_workload(repo, data, name, spec)
    base['prepared_phase_supply'] = prep['supply']
    cell['readiness']['requirements'] = [{'kind':'preparation', **prep['supply']}]
    for path, doc in ((bp,base),(ep,exp)):
        path.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    cfg = compose(repo, spec['model_fragment'], cell['dataset_fragment'], name, name)
    ph = protocol_hash(cfg.config_hash, cfg.values['arm'], spec['split_identity_hash'])
    auth = 'docs/decisions/2026-11-02-owner-real100-concurrency.md'
    if not (repo/auth).is_file(): raise ValueError('recorded owner scope missing')
    spec.update(composed_config_hash=cfg.config_hash, protocol_hash=ph, authority=auth,
                decision_id=name+'-owner-admission', decision_relpath=f'coordination/{name}/root-admission-v1.json')
    protocol = {'schema':'saea-real-engine-protocol-v1','cells':{'arm':spec},
                'profile':spec['profile'],'profile_sha256':spec['profile_sha256'],
                'prepared_phase_supply':prep['supply'],'rows':100,'claims_benefit':False,
                'inference':'Fresh real audio P feeds fresh S; no historical response replay',
                'runtime_receipt_sha256':spec['runtime_receipt_sha256']}
    docs = {'candidate.json':spec, 'protocol-registered-v1.json':protocol}
    # Use precisely the bytes write() will emit when constructing references.
    encoded = lambda doc: (json.dumps(doc,indent=2,sort_keys=True)+'\n').encode()
    protocol_pin = {'relpath':str((pkg/'protocol-registered-v1.json').relative_to(repo)),
                    'sha256':hashlib.sha256(encoded(protocol)).hexdigest()}
    rows=[]
    for kind,rid,aid,calls in [('generation',run,'generation-v1',400),('fit',run+'-FIT','fit-v1',200)]:
        budget={**exp['execution_plan'],'planned_model_calls':calls}
        plan=ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',
                           split_identity_hash=spec['split_identity_hash'],protocol_hash=ph,**budget)
        plan.validate(source_root=repo)
        doc={'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':name,'experiment':name,
             'composed_config_hash':cfg.config_hash,'expected_session_id':spec['service_session'],
             'execution_plan':asdict(plan),'prepared_phase_supply':prep['supply'],
             'protocol_document':protocol_pin,'status':'held'}
        filename=f'{kind}-plan-registered-v1.json';docs[filename]=doc
        rows.append(['2026-11-02',rid,'bounded-discovery-probe','Fixed real100 concurrency '+kind,
                     spec['model_fragment'],'slurp TRAIN fixed100','discovery',spec['split_identity_hash'],'no',
                     str(calls),str(budget['planned_speech_audio_seconds']),str(budget['planned_gpu_hours']),
                     'Historical costs preserved',str((pkg/filename).relative_to(repo)),
                     f'Plan SHA256 {hashlib.sha256(encoded(doc)).hexdigest()}; root records existing owner authorization under {auth}; held'])
    docs['analysis-recipe-v1.json']={'schema':'saea-real-engine-analysis-v1','rows':100,
        'run_ids':{'generation':run,'fit':run+'-FIT','read':run+'-READ'},
        'test_contact_authorized':False,'gold_in_runtime':False,'claims_benefit':False,
        'metrics':['SLU-F1','intent accuracy','complete rows','wall seconds per row','per-card task utilization',
                   'queue and phase latency','token and audio cost'],
        'decision':'Compare the entire fixed population, including failures; no fastest-finished subset'}
    for rel, doc in docs.items(): write(pkg/rel,doc)
    with ledger.open('a') as f:
        f.write('\n## '+name+'\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n')
    (repo/'docs/readiness'/f'{name}-preregistration.md').write_text(
        '# Fixed real100 concurrency comparison\n\nSame 100 real TRAIN rows across levels. Fresh P feeds S, normal 200 generation contacts, cap400 including fallback; 200 FIT contacts. Native journals and checkpoints preserve known responses and costs; unknown calls must be reconciled before continuation. Shared admission and clean-tree preflight precede detached dispatch. Report complete population, latency, allocated resources and measured utilization separately. No test tuning.\n')
    return {'registered':True,'rows':100,'new_model_calls':0,'generation_authorized':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage', 'prepare', 'register'])
    for key in ('repo', 'data', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('source', 'manifest', 'roster', 'parent-view'):
        parser.add_argument('--'+key, type=Path)
    parser.add_argument('--level', type=int, default=32)
    parser.add_argument('--profile')
    parser.add_argument('--session', default='serving-tfgrpo-ctx48k-resident-v1')
    a = parser.parse_args()
    if a.mode == 'register': result = register(a.repo, a.data, a.out)
    elif a.mode == 'prepare': result = prepare(a.repo, a.data, a.out)
    else: result = stage(a.repo, a.data, a.source, a.manifest, a.roster, a.parent_view, a.level, a.profile, a.session, a.out)
    print(json.dumps(result, sort_keys=True))
