"""Stage fresh held odd40 halves and prepare their real first-phase supply.

No registration, readiness promotion, tokenizer contact or generation. All writes
are create-only; an interrupted preparation must be inspected before continuation.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).parent)]
import tfrl_odd40_candidates as candidates
from speech_aware_evidence_acquisition.core.atomic_file import publish_new


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def confined(root, rel):
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('path escapes its declared root')
    return path


def identity(half):
    if half not in ('h1', 'h2'):
        raise ValueError('fixed construction half required')
    return (f'slurp-tfrl-odd40-i1-{half}-v1',
            f'SAEA-SLURP-TFRL-ODD40-I1-{half.upper()}-V1',
            313 if half == 'h1' else 312)


def require_unspent(data, run, *, preparation=False):
    for rid in (run, run + '-FIT'):
        for kind in ('runs', 'attempts', 'prepared') if preparation else ('runs', 'attempts'):
            if (data / kind / rid).exists():
                raise ValueError('existing run/FIT identity; inspect before continuation')


def derive(repo, original, experiment, roster, half):
    name, run, count = identity(half)
    ids = [row['sample_id'] for row in roster['items']]
    population, = experiment['populations']
    if (len(ids) != count or len(set(ids)) != count or ids != population['ids']
            or roster['split_identity_hash'] != experiment['population']['split_identity_hash']):
        raise ValueError('exact original construction half required')
    seconds = [row['audio_seconds'] for row in roster['items']]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError('finite positive actual audio duration required')
    base, proof = candidates.derive_method(repo, original)
    fit = copy.deepcopy(original['native_contact_fit'])
    fit.update(run_id=run + '-FIT', attempt_id='fit-v1', contact_cap=count * proof['fit_cap_per_row'])
    base['native_contact_fit'] = fit
    base['what_this_is'] = 'Held odd40 whole-answer construction with shared P, B, A/A, A and conditional choice; fresh supply and admission pending.'
    exp = copy.deepcopy(experiment)
    cell, = exp['cells']
    cell.update(cell_id=name, baseline_fragment=name, run_id=run, attempt_id='generation-v1',
                allowed_execution_stages=['check'], readiness={'state': 'held', 'requirements': [],
                    'reason': 'Own supply, real FIT, partial-contact recovery and registration/admission pending.'})
    # A spent source session is not evidence of the current service identity.
    exp.pop('service_session', None)
    exp.update(experiment_id=run, purpose=base['what_this_is'])
    exp['execution_plan'].update(planned_model_calls=count * proof['fallback_inclusive_cap_per_row'],
        planned_speech_audio_seconds=math.ceil(math.fsum(seconds) * proof['fallback_inclusive_cap_per_row']))
    exp['stage3_algorithm'] = {'id': 'shared-rollout', 'operators_used': ['rollout', 'scoring'],
        'note': 'Shared P; odd-rank top40 evidence B, original A/A and A; conditional whole-answer A/B choice.'}
    exp['stage3_measurement'].update(rows=count, claims_benefit=False, expected_effect_pp=0,
        measurement_design='material-source', read_roster=roster['roster_id'], tuning_roster='none',
        note='Complete625 construction gate: SLU-F1 >= +1pp versus same-trajectory A, Intent non-decreasing; no system claim.')
    return name, base, exp, proof


def stage(repo, source, data, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment()
    head, source_head = require_clean_study_repo(repo), require_clean_study_repo(source)
    if out.exists() or not out.resolve().is_relative_to(data.resolve() / 'work/codex'):
        raise ValueError('fresh owned staging output required')
    outputs, source_pins, data_pins, dependency_pins, summaries, seen = {}, {}, {}, {}, [], set()

    def source_doc(rel):
        path = confined(source, rel)
        source_pins[rel] = sha(path)
        return read(path)

    def dependency(rel):
        src, dst = confined(source, rel), confined(repo, rel)
        blob = src.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        source_pins[rel] = digest
        dependency_pins[rel] = digest
        if dst.exists():
            if dst.read_bytes() != blob:
                raise ValueError('destination dependency differs: ' + rel)
        else:
            outputs[rel] = blob

    for half in ('h1', 'h2'):
        old = f'slurp-tfrl-current-seed-i1-{half}-v1'
        original = source_doc(f'configs/baseline/{old}.json')
        experiment = source_doc(f'configs/experiment/{old}.json')
        roster_rel = experiment['population']['roster_receipt']
        roster = source_doc(roster_rel)
        name, base, exp, proof = derive(repo, original, experiment, roster, half)
        cell, = exp['cells']
        run = cell['run_id']
        require_unspent(data, run, preparation=True)
        ids = exp['populations'][0]['ids']
        if set(ids) & seen:
            raise ValueError('overlapping construction halves')
        seen.update(ids)
        dataset_rel = f"configs/dataset/{cell['dataset_fragment']}.json"
        view = source_doc(dataset_rel)
        items = {r['sample_id']: r for r in view['items']}
        if len(items) != 625 or len(view['items']) != 625 or not set(ids) <= items.keys():
            raise ValueError('full unique real625 dataset required')
        roster_seconds = {r['sample_id']: r['audio_seconds'] for r in roster['items']}
        for sid in ids:
            row = items[sid]
            if row['audio_seconds'] != roster_seconds[sid]:
                raise ValueError('roster and recorded media duration differ')
            if (not sid.startswith('slurp:train:') or '/slurp_real/' not in row['media_relpath']
                    or not row['audio_sha256'] or not confined(data, row['media_relpath']).is_file()):
                raise ValueError('real training media required')
        for pin in base['preparation_source_receipts']:
            if pin['root'] == 'repository':
                dependency(pin['relpath'])
                if dependency_pins[pin['relpath']] != pin['sha256']:
                    raise ValueError('repository preparation pin differs')
            elif pin['root'] == 'data':
                if sha(confined(data, pin['relpath'])) != pin['sha256']:
                    raise ValueError('data preparation pin differs')
                data_pins[pin['relpath']] = pin['sha256']
            else:
                raise ValueError('unknown preparation root')
        model_rel = f"configs/model/{cell['model_fragment']}.json"
        model = source_doc(model_rel)
        for rel in (roster_rel, dataset_rel, model_rel, 'docs/' + model['runtime_receipt']):
            dependency(rel)
        for kind, doc in (('baseline', base), ('experiment', exp)):
            rel = f'configs/{kind}/{name}.json'
            if (repo / rel).exists():
                raise ValueError('fresh odd40 configuration required')
            outputs[rel] = raw(doc)
        summaries.append({'half': half, 'cell': name, 'run_id': run, 'rows': len(ids),
            'normal_calls_min': len(ids) * 4, 'normal_calls_max': len(ids) * 5,
            'fallback_cap': len(ids) * 10, 'fit_cap': len(ids) * 5, 'proof': proof})
    if len(seen) != 625:
        raise ValueError('complete625 union required')
    if require_clean_study_repo(repo) != head or require_clean_study_repo(source) != source_head:
        raise ValueError('source commit changed')
    for rel, digest in source_pins.items():
        if sha(confined(source, rel)) != digest:
            raise ValueError('source bytes changed')
    result = {'schema': 'saea-odd40-held-cells-v1', 'source_commit': source_head,
        'implementation_commit': head, 'source_hashes': source_pins, 'data_hashes': data_pins,
        'dependency_hashes': dependency_pins, 'rows': 625, 'halves': summaries,
        'output_hashes': {p: hashlib.sha256(b).hexdigest() for p, b in outputs.items()},
        'model_calls': 0, 'tokenizer_calls': 0, 'readiness': 'held', 'registered': False,
        'live_service_verified': False, 'generation_authorized': False}
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', raw(result))
    for rel, blob in outputs.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_new(target, blob)
        target.chmod(0o644)
    publish_new(out / 'complete.json', raw(result))
    return result


def prepare(repo, data, derivation, lock, out, half):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment()
    head = require_clean_study_repo(repo)
    d = read(derivation)
    if (d.get('schema') != 'saea-odd40-held-cells-v1' or d.get('rows') != 625
            or d.get('model_calls') != 0 or d.get('readiness') != 'held'
            or d.get('registered') is not False or d.get('generation_authorized') is not False):
        raise ValueError('completed held odd40 derivation required')
    pins = {**d['dependency_hashes'], **d['output_hashes']}
    dp, lp = sha(derivation), sha(lock)

    def verify_pins():
        for rel, digest in pins.items():
            if sha(confined(repo, rel)) != digest:
                raise ValueError('derived configuration or dependency changed')
        for rel, digest in d['data_hashes'].items():
            if sha(confined(data, rel)) != digest:
                raise ValueError('preparation data changed')
        if sha(derivation) != dp or sha(lock) != lp:
            raise ValueError('derivation or dataset lock changed')

    verify_pins()
    name, run, count = identity(half)
    require_unspent(data, run, preparation=True)
    base = read(repo / f'configs/baseline/{name}.json')
    exp = read(repo / f'configs/experiment/{name}.json')
    cell, = exp['cells']
    ids = exp['populations'][0]['ids']
    if (cell['cell_id'] != name or cell['run_id'] != run or cell['baseline_fragment'] != name
            or cell['readiness'] != {'state': 'held', 'requirements': [],
                'reason': 'Own supply, real FIT, partial-contact recovery and registration/admission pending.'}
            or cell['allowed_execution_stages'] != ['check'] or len(ids) != count or len(set(ids)) != count
            or 'prepared_phase_supply' in base or base['native_contact_fit']['run_id'] != run + '-FIT'
            or base['native_contact_fit']['contact_cap'] != 5 * count):
        raise ValueError('exact fresh held odd40 half required')
    candidates.cm.verify_declared_module(base['native_episode']['consumption_module'], base, repo_root=repo)
    target = data / 'prepared' / run / 'phase-supply-v1.json'
    if out.exists() or not out.resolve().is_relative_to(data.resolve() / 'work/codex'):
        raise ValueError('fresh owned preparation output required')
    out.mkdir(parents=True)
    record = {'schema': 'saea-odd40-supply-v1', 'source_commit': head, 'rows': count,
        'cell': name, 'run_id': run, 'source_hashes': pins, 'data_hashes': d['data_hashes'],
        'derivation_sha256': dp, 'dataset_lock_sha256': lp, 'model_calls': 0,
        'tokenizer_calls': 0, 'readiness': 'held', 'registered': False, 'automatic_replay': False}
    publish_new(out / 'intent.json', raw(record))
    try:
        runtime = build_flight_runtime(repo, data, model=cell['model_fragment'], dataset=cell['dataset_fragment'],
            baseline=name, experiment=name, run_id=run, attempt_id=cell['attempt_id'],
            runtime_root=data / 'work/scratch/runtime-root', lock_path=lock)
        runtime.prepare_phase_supply(target, max_workers=16)
        verify_output(read(target)['output'], ids)
        verify_pins()
        if require_clean_study_repo(repo) != head:
            raise ValueError('implementation changed during preparation')
        result = {**record, 'supply': {'relpath': str(target.relative_to(data)), 'sha256': sha(target)}}
        publish_new(out / 'complete.json', raw(result))
        return result
    except BaseException as error:
        publish_new(out / 'failed.json', raw({'error': str(error), 'automatic_replay': False}))
        raise


def bind(repo, data, derivation, lock, preparations, out):
    """Bind both completed supplies and verify the shared runtime identity/seed."""
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment()
    head = require_clean_study_repo(repo)
    if out.exists() or not out.resolve().is_relative_to(data.resolve() / 'work/codex'):
        raise ValueError('fresh owned binding output required')
    if len(preparations) != 2:
        raise ValueError('both completed odd40 supplies required')
    d = read(derivation)
    if d.get('schema') != 'saea-odd40-held-cells-v1' or d.get('rows') != 625:
        raise ValueError('exact odd40 derivation required')
    pins = {**d['dependency_hashes'], **d['output_hashes']}
    for rel, digest in pins.items():
        if sha(confined(repo, rel)) != digest:
            raise ValueError('derived inputs changed before binding')
    pending, seen = [], set()
    for path in preparations:
        prep = read(path)
        if prep['rows'] not in (313, 312) or prep['rows'] in seen:
            raise ValueError('distinct complete construction halves required')
        seen.add(prep['rows'])
        half = 'h1' if prep['rows'] == 313 else 'h2'
        name, run, count = identity(half)
        require_unspent(data, run)
        if (prep['schema'] != 'saea-odd40-supply-v1' or prep['cell'] != name or prep['run_id'] != run
                or prep['derivation_sha256'] != sha(derivation) or prep['dataset_lock_sha256'] != sha(lock)
                or prep['source_hashes'] != pins or prep['data_hashes'] != d['data_hashes']
                or prep['model_calls'] != 0 or prep['tokenizer_calls'] != 0
                or prep['readiness'] != 'held' or prep['registered'] is not False
                or prep['supply']['relpath'] != f'prepared/{run}/phase-supply-v1.json'):
            raise ValueError('completed owned model-free odd40 supply required')
        for rel, digest in prep['data_hashes'].items():
            if sha(confined(data, rel)) != digest:
                raise ValueError('preparation data changed before binding')
        target = confined(data, prep['supply']['relpath'])
        if sha(target) != prep['supply']['sha256']:
            raise ValueError('prepared supply bytes changed')
        bp, ep = (repo / f'configs/{kind}/{name}.json' for kind in ('baseline', 'experiment'))
        base, exp = read(bp), read(ep)
        verify_output(read(target)['output'], exp['populations'][0]['ids'])
        if ('prepared_phase_supply' in base or exp['cells'][0]['readiness']['state'] != 'held'
                or exp['cells'][0]['readiness']['requirements']
                or exp['cells'][0]['allowed_execution_stages'] != ['check']):
            raise ValueError('unbound held half required')
        base['prepared_phase_supply'] = prep['supply']
        exp['cells'][0]['readiness']['requirements'] = [{'kind': 'preparation', **prep['supply']}]
        pending.append((prep, bp, ep, base, exp))
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', raw({'source_commit': head,
        'preparation_hashes': [sha(p) for p in preparations], 'model_calls': 0}))
    for prep, bp, ep, base, exp in pending:
        for path, doc in ((bp, base), (ep, exp)):
            publish_new(out / (path.stem + '.' + path.parent.name + '.before.json'), path.read_bytes())
            path.write_bytes(raw(doc))
    try:
        checks, outputs = [], {}
        for prep, bp, ep, base, exp in pending:
            cell, = exp['cells']
            runtime = build_flight_runtime(repo, data, model=cell['model_fragment'], dataset=cell['dataset_fragment'],
                baseline=cell['baseline_fragment'], experiment=cell['cell_id'], run_id=cell['run_id'],
                attempt_id=cell['attempt_id'], runtime_root=data / 'work/scratch/runtime-root', lock_path=lock)
            verified = runtime.verify_prepared_artifacts()
            if not verified or not all(v['identity_ok'] and v.get('seed_ok', False) for v in verified):
                raise ValueError('shared runtime prepared identity/seed verification failed')
            checks.append({'cell': cell['cell_id'], 'rows': prep['rows'], 'checks': verified, 'supply': prep['supply']})
            outputs.update({str(p.relative_to(repo)): sha(p) for p in (bp, ep)})
        result = {'schema': 'saea-odd40-supply-binding-v1', 'rows': 625, 'source_commit': head,
            'halves': checks, 'output_hashes': outputs, 'model_calls': 0, 'tokenizer_calls': 0,
            'readiness': 'held', 'registered': False, 'generation_authorized': False}
        publish_new(out / 'complete.json', raw(result))
        return result
    except BaseException as error:
        publish_new(out / 'failed.json', raw({'error': str(error), 'automatic_replay': False}))
        raise


def verify_output(output, ids):
    if set(output) != {'facets', 'values'} or set(output['values']) != {'perception'}:
        raise ValueError('exact first-phase supply shape required')
    for phases in output.values():
        if phases and (set(phases) != {'perception'} or set(phases['perception']) != set(ids)):
            raise ValueError('incomplete or foreign first-phase supply rows')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['stage', 'prepare', 'bind'])
    for key in ('repo', 'data', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('source', 'derivation', 'lock'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--half', choices=['h1', 'h2'])
    p.add_argument('--preparation', type=Path, action='append')
    a = p.parse_args()
    if a.action == 'stage':
        if not a.source: p.error('stage requires --source')
        result = stage(a.repo, a.source, a.data, a.out)
    elif a.action == 'bind':
        if not all((a.derivation, a.lock, a.preparation)): p.error('bind requires --derivation, --lock and --preparation')
        result = bind(a.repo, a.data, a.derivation, a.lock, a.preparation, a.out)
    else:
        if not all((a.derivation, a.lock, a.half)): p.error('prepare requires --derivation, --lock and --half')
        result = prepare(a.repo, a.data, a.derivation, a.lock, a.out, a.half)
    print(json.dumps(result))
