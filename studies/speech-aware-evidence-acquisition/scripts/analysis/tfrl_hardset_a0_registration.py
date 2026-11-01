"""Register fixed439 independent A0 observations using the shared flight harness.

Stage binds existing first-P values, never resamples data. Registration remains
held; existing admission repins changed source identity with output equality.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from tfrl_hardset_a0_workload import NAME, RUN, read, raw, sha, confined
from tfrl_current_generation import MODEL, SESSION, PROFILE, RECEIPT, LOCK, LOCK_SHA
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
AUTH = 'docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'


def pin(root, rel):
    return {'relpath': rel, 'sha256': sha(confined(root, rel))}


def pinned(root, binding):
    path = confined(root, binding['relpath'])
    if sha(path) != binding['sha256']:
        raise ValueError('pinned artifact changed: '+binding['relpath'])
    return read(path)


def require_unspent(repo, data):
    ledger = (repo/'docs/exposure-ledger.md').read_text()
    for run in (RUN, RUN+'-FIT', RUN+'-READ'):
        if '| '+run+' |' in ledger or any((data/k/run).exists() for k in ('attempts', 'runs')):
            raise ValueError('existing registration or spend: inspect before continuation')


def scientific_hash(base):
    value = copy.deepcopy(base)
    value.pop('prepared_phase_supply', None)
    return hashlib.sha256(raw(value)).hexdigest()


def stage(repo, data, prepared, derivation, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    from tfrl_odd40_cells import verify_output
    assert_execution_environment()
    head = require_clean_study_repo(repo)
    require_unspent(repo, data)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned stage output required')
    d, prep = read(derivation), read(prepared)
    if (d['schema'] != 'saea-hardset-a0-held-workload-v1' or d['rows'] != 439
            or prep['schema'] != 'saea-hardset-a0-first-p-supply-v1' or prep['rows'] != 439
            or prep['derivation_sha256'] != sha(derivation) or prep['run_id'] != RUN
            or prep['generation_authorized'] is not False or prep['new_model_calls'] != 0
            or prep['new_tokenizer_calls'] != 0):
        raise ValueError('completed fixed439 model-free preparation required')
    for rel, digest in d['output_hashes'].items():
        if sha(confined(repo, rel)) != digest:
            raise ValueError('held workload input changed: '+rel)
    bp, ep = [repo/f'configs/{kind}/{NAME}.json' for kind in ('baseline', 'experiment')]
    base, exp = read(bp), read(ep)
    c, = exp['cells']
    if (c['readiness']['state'] != 'held' or c['allowed_execution_stages'] != ['check']
            or 'prepared_phase_supply' in base or c['model_fragment'] != MODEL
            or read(repo/f'configs/model/{MODEL}.json')['runtime_receipt'] != RECEIPT):
        raise ValueError('unbound held selected-engine workload required')
    if not native_contact_custody_enabled({**base, 'carrier_lock_key': 'slurp', 'transport': 'vllm'}):
        raise ValueError('A0 native contact recovery must be integrated before staging')
    supply = pinned(data, prep['supply'])
    verify_output(supply['output'], exp['populations'][0]['ids'])
    base['prepared_phase_supply'] = prep['supply']
    c['readiness']['requirements'] = [{'kind': 'preparation', **prep['supply']}]
    c['allowed_execution_stages'] = ['check', 'prepare-supply', 'worker', 'campaign']
    exp['service_session'] = SESSION
    out.mkdir(parents=True)
    publish_new(out/'intent.json', raw({'source_commit': head, 'new_model_calls': 0}))
    for path, doc in ((bp, base), (ep, exp)):
        publish_new(out/(path.parent.name+'.before.json'), path.read_bytes())
        path.write_bytes(raw(doc))
    record = {'schema': 'saea-hardset-a0-runtime-stage-v1', 'rows': 439,
              'prepared_sha256': sha(prepared), 'derivation_sha256': sha(derivation),
              'source_commit': head, 'readiness': 'held', 'new_model_calls': 0,
              'output_hashes': {str(p.relative_to(repo)): sha(p) for p in (bp, ep)},
              'supply': prep['supply'], 'output_values_reused': True,
              'remaining': ['commit stage', 'register', 'commit registration',
                            'shared admission and output-preserving repin', 'commit',
                            'shared preflight', 'detached dispatch']}
    publish_new(out/'complete.json', raw(record))
    return record


def validate(repo, data, name, spec, exp, view):
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    from tfrl_odd40_cells import verify_output
    c, = exp['cells']; pop, = exp['populations']; summary = exp['population']
    if (name != NAME or spec['run_id'] != RUN or spec['attempt_id'] != 'generation-v1'
            or c['run_id'] != RUN or c['attempt_id'] != spec['attempt_id']
            or c['baseline_fragment'] != NAME or c['model_fragment'] != MODEL
            or spec['fit_run_id'] != RUN+'-FIT' or spec['fit_attempt_id'] != 'fit-v1'
            or exp['service_session'] != SESSION or spec['service_session'] != SESSION
            or spec['profile'] != PROFILE or spec['runtime_receipt'] != RECEIPT):
        raise ValueError('exact fixed439 A0 runtime identity required')
    roster = pinned(repo, spec['roster'])
    ids = [r['sample_id'] for r in roster['items']]
    if (len(ids) != 439 or len(set(ids)) != 439 or pop['ids'] != ids
            or summary['count'] != 439 or spec['rows'] != 439
            or len(view['items']) != 625 or view['count'] != 625
            or pinned(repo, spec['dataset']) != view
            or len({r['sample_id'] for r in view['items']}) != 625
            or not set(ids) <= {r['sample_id'] for r in view['items']}
            or len({split_identity_hash(prefixed('slurp', ids)), roster['split_identity_hash'],
                    summary['split_identity_hash'], spec['split_identity_hash']}) != 1
            or pop['dataset_fragment'] != c['dataset_fragment']
            or summary['dataset_fragment'] != c['dataset_fragment']
            or summary['roster_receipt'] != spec['roster']['relpath']):
        raise ValueError('fixed439 subset of unchanged625 data required')
    byid = {r['sample_id']: r for r in view['items']}
    for r in roster['items']:
        v = byid[r['sample_id']]
        if (not r['sample_id'].startswith('slurp:train:') or '/slurp_real/' not in v['media_relpath']
                or len(v['audio_sha256']) != 64 or v['audio_seconds'] != r['audio_seconds']):
            raise ValueError('real audio custody changed')
    base = read(repo/f'configs/baseline/{NAME}.json')
    fit = base['native_contact_fit']
    if (scientific_hash(base) != spec['scientific_baseline_sha256']
            or not native_contact_custody_enabled({**base, 'carrier_lock_key': 'slurp', 'transport': 'vllm'})
            or fit['contact_cap'] != 878 or fit['run_id'] != RUN+'-FIT' or fit['attempt_id'] != 'fit-v1'
            or exp['execution_plan']['planned_model_calls'] != 1756
            or exp['execution_plan']['planned_speech_audio_seconds'] < math.ceil(4*sum(r['audio_seconds'] for r in roster['items']))
            or exp['stage3_algorithm']['id'] != 'shared-rollout'
            or exp['stage3_algorithm']['operators_used'] != ['rollout', 'scoring']
            or exp['stage3_measurement']['claims_benefit'] is not False):
        raise ValueError('registered two-contact method or budget changed')
    verify_output(pinned(data, base['prepared_phase_supply'])['output'], ids)
    if spec['gain_lock'] != {'relpath': LOCK, 'sha256': LOCK_SHA} or pinned(repo, spec['gain_lock'])['status'] != 'locked_full_train_fold_b':
        raise ValueError('locked first-system gain required')
    return 439


def register(repo, data, stage_record):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment, ExecutionPlan
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
    assert_execution_environment(); head = require_clean_study_repo(repo)
    require_unspent(repo, data)
    staged = read(stage_record)
    if staged['schema'] != 'saea-hardset-a0-runtime-stage-v1' or staged['rows'] != 439:
        raise ValueError('complete held runtime stage required')
    for rel, digest in staged['output_hashes'].items():
        if sha(confined(repo, rel)) != digest:
            raise ValueError('staged input changed')
    exp = read(repo/f'configs/experiment/{NAME}.json'); c, = exp['cells']
    base = read(repo/f'configs/baseline/{NAME}.json'); split = exp['population']['split_identity_hash']
    pkg = repo/'docs/readiness'/NAME
    if pkg.exists() or c['readiness']['state'] != 'held':
        raise ValueError('fresh held registration required')
    cfg = compose(repo, MODEL, c['dataset_fragment'], NAME, NAME)
    ph = protocol_hash(cfg.config_hash, cfg.values['arm'], split)
    identity = {'cell': NAME, 'experiment': NAME, 'run_id': RUN, 'attempt_id': c['attempt_id'],
                'fit_run_id': RUN+'-FIT', 'fit_attempt_id': 'fit-v1', 'rows': 439,
                'composed_config_hash': cfg.config_hash, 'protocol_hash': ph, 'split_identity_hash': split}
    spec = {**identity, 'workload_kind': 'tfrl-hardset-a0-observation', 'model_fragment': MODEL,
            'baseline': NAME, 'service_session': SESSION, 'profile': PROFILE, 'runtime_receipt': RECEIPT,
            'authority': AUTH, 'decision_id': NAME+'-owner-admission',
            'decision_relpath': f'coordination/{NAME}/root-admission-v1.json',
            'roster': pin(repo, exp['population']['roster_receipt']),
            'dataset': pin(repo, f"configs/dataset/{c['dataset_fragment']}.json"),
            'gain_lock': pin(repo, LOCK), 'scientific_baseline_sha256': scientific_hash(base),
            'stage_sha256': sha(stage_record),
            'generation_program': 'Fresh P then S, fixed observed-error subset; no precision gain claim.'}
    validate(repo, data, NAME, spec, exp, read(repo/spec['dataset']['relpath']))
    protocol = {'schema': 'saea-hardset-a0-generation-v1', 'cells': {'arm': identity},
                'profile': PROFILE, 'roster': spec['roster'], 'gain_lock': spec['gain_lock'],
                'prepared_phase_supply': base['prepared_phase_supply'], 'normal_generation_calls': 878,
                'fallback_inclusive_cap': 1756, 'claims_benefit': False,
                'budget_basis': '16 allocated card-hours is a cap, not a performance prediction.'}
    docs = {'candidate.json': spec, 'protocol-registered-v1.json': protocol}
    protocol_pin = {'relpath': str((pkg/'protocol-registered-v1.json').relative_to(repo)),
                    'sha256': hashlib.sha256(raw(protocol)).hexdigest()}
    ledger_rows = []
    for kind, rid, aid, calls in [('generation', RUN, 'generation-v1', 1756), ('fit', RUN+'-FIT', 'fit-v1', 878)]:
        budget = {**exp['execution_plan'], 'planned_model_calls': calls}
        ExecutionPlan(run_id=rid, attempt_id=aid, carrier_lock_key='slurp', split_identity_hash=split,
                      protocol_hash=ph, **budget).validate(source_root=repo)
        plan = {'schema': f'saea-registered-{kind}-plan-v1', 'plan_kind': kind, 'cell': NAME, 'experiment': NAME,
                'composed_config_hash': cfg.config_hash, 'expected_session_id': SESSION,
                'execution_plan': {**budget, 'schema': 'saea-execution-plan-v1', 'study': 'speech-aware-evidence-acquisition',
                    'run_id': rid, 'attempt_id': aid, 'carrier_lock_key': 'slurp',
                    'split_identity_hash': split, 'protocol_hash': ph},
                'prepared_phase_supply': base['prepared_phase_supply'], 'protocol_document': protocol_pin,
                'status': 'Registered before contact; held pending shared repin/admission/preflight.'}
        rel = f'{kind}-plan-registered-v1.json'; docs[rel] = plan
        ledger_rows.append(['2026-11-01', rid, 'bounded-discovery-probe', 'Fixed439 A0 '+kind,
            MODEL, 'slurp TRAIN isolated error candidates', 'discovery', split, 'no', str(calls),
            str(budget['planned_speech_audio_seconds']), str(budget['planned_gpu_hours']),
            'Historical errors and costs retained; one observation is not statistical hardness confirmation',
            str((pkg/rel).relative_to(repo)), f'Plan SHA256 {hashlib.sha256(raw(plan)).hexdigest()}; root countersigned under {AUTH}; held; no test tuning'])
    docs['analysis-recipe-v1.json'] = {'schema': 'saea-hardset-a0-read-v1', 'rows': 439,
        'generation_run_id': RUN, 'attempt_id': 'generation-v1', 'roster': spec['roster'],
        'official_scoring_jobs': 1, 'new_model_calls': 0, 'claims_benefit': False,
        'scope': 'Score all439 sealed final answers once; join prior evidence by sample ID; no subset precision claim.',
        'execution': 'Shared campaign scores after generation lease release; complete response and cost custody retained.'}
    pkg.mkdir(parents=True)
    for rel, doc in docs.items(): publish_new(pkg/rel, raw(doc))
    ledger = repo/'docs/exposure-ledger.md'
    ledger.write_text(ledger.read_text()+'\n## Fixed439 independent A0 observation\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in ledger_rows)+'\n')
    publish_new(repo/'docs/readiness'/f'{NAME}-preregistration.md', b'# Fixed439 independent A0 observation\n\nReuse the fixed625 data view and exact439 error-candidate roster. Fresh P/S,878 normal contacts,1756 fallback-inclusive generation cap,878 FIT cap. Retain prior exposure. No statistical-hardness or improvement claim. Shared output-preserving repin, live resident verification, admission and preflight precede detached dispatch.\n')
    return {'cell': NAME, 'rows': 439, 'source_commit': head, 'new_model_calls': 0, 'readiness': 'held'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage', 'register'])
    for key in ('repo', 'data'): parser.add_argument('--'+key, type=Path, required=True)
    for key in ('prepared', 'derivation', 'out', 'stage-record'): parser.add_argument('--'+key, type=Path)
    a = vars(parser.parse_args()); mode = a.pop('mode')
    if mode == 'stage': a.pop('stage_record'); result = stage(**a)
    else:
        for k in ('prepared', 'derivation', 'out'): a.pop(k)
        result = register(**a)
    print(json.dumps(result, sort_keys=True))
