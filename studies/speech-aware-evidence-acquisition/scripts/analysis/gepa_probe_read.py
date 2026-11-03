"""Read a finalized shared-P GEPA mechanism probe through the official scorer.

This is not search acceptance or evidence of stable improvement. Both complete
Solver arms are scored; the original delivering lock must match sealed scores.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from tfrl_odd40_cells import read, raw, sha
from tfrl_hardset_a0_registration import pin, pinned
from tfrl_current_half_result import score_table
from h1_thinking_comparison import native_costs
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

ARMS = ('gepa', 'lock')


def summarize(results, items):
    ids = [r['sample_id'] for r in items]
    if len(ids) != 16 or len(set(ids)) != 16 or set(results) != set(ARMS):
        raise ValueError('complete fixed16 gepa/lock mechanism probe required')
    tables = {arm: score_table(results[arm], ids) for arm in ARMS}
    identity_keys = ('evaluator_commit', 'scorer_version', 'split_id', 'projection', 'filler_normalization')
    identity = lambda arm: {k: results[arm]['aggregate'][k] for k in identity_keys}
    if identity('gepa') != identity('lock'):
        raise ValueError('official scorer identities differ')
    success = {}
    for arm in ARMS:
        success[arm] = {}
        for sid, row in tables[arm].items():
            values = [row[k] for k in ('slot_f1_exact_fp', 'slot_f1_exact_fn')]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or int(v) != v for v in values):
                raise ValueError('finite nonnegative slot counts required')
            success[arm][sid] = int(row['scenario_action_exact'] == 1 and values == [0, 0])
    return {'rows': len(ids), 'official': {a: results[a]['aggregate']['official'] for a in ARMS},
        'strict_success': {a: sum(success[a].values()) for a in ARMS},
        'task_repairs': sum(success['gepa'][s] > success['lock'][s] for s in ids),
        'task_breaks': sum(success['gepa'][s] < success['lock'][s] for s in ids),
        'candidate_accepted': False, 'claims_stable_gain': False,
        'purpose': 'Fixed16 native mechanism check, not an algorithm quality verdict.'}


def verify_delivered_scores(lock_result, delivered):
    from stage3_rollout_score import correctness_fields
    rows = lock_result['per_sample']
    if len(rows) != len(delivered) or {r['sample_id'] for r in rows} != set(delivered):
        raise ValueError('lock scores differ from sealed delivering population')
    for row in rows:
        if correctness_fields(row) != correctness_fields(delivered[row['sample_id']]):
            raise ValueError('lock scores differ from sealed delivering scores')


def validate_recipe(recipe, spec):
    expected = {'schema': 'saea-gepa-probe-read-v1', 'rows': 16,
        'generation_run_id': spec['run_id'], 'attempt_id': spec['attempt_id'],
        'arms': list(ARMS), 'delivered_arm': 'lock', 'roster': spec['roster']}
    if any(recipe.get(k) != v for k, v in expected.items()) or spec['rows'] != 16:
        raise ValueError('exact preregistered fixed16 GEPA read recipe required')


def run(repo, data, out, name):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    from registered_workload import validate_workload
    from tfgrpo_h2_read import registered_scorer
    import stage3_rollout_score as scorer
    assert_execution_environment()
    if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
        raise ValueError('confined candidate name required')
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('owned analysis output required')
    pkg = repo/'docs/readiness'/name
    spec = read(pkg/'candidate.json')
    if spec.get('workload_kind') != 'gepa-solver-template-probe':
        raise ValueError('registered GEPA mechanism probe required')
    validate_workload(repo, data, name, spec)
    recipe = read(pkg/'analysis-recipe-v1.json')
    validate_recipe(recipe, spec)
    rid, attempt = spec['run_id'], spec['attempt_id']
    directory = data/'runs'/rid/attempt
    bundle = verify_bundle(directory/(rid+'.run-manifest.json'))
    roster = pinned(repo, spec['roster']); ids = [r['sample_id'] for r in roster['items']]
    if bundle.attempt_state != 'FINALIZED' or len(bundle.split_ids) != 16 or set(bundle.split_ids) != set(ids):
        raise ValueError('complete finalized fixed16 required')
    fragments = bundle.manifest['config']['fragments']
    cfg = compose(repo, *(fragments[k] for k in ('model', 'dataset', 'baseline', 'experiment')))
    if cfg.config_hash != bundle.manifest['config']['config_hash']:
        raise ValueError('sealed probe configuration changed')
    roles = cfg.values['native_episode']['roles']
    group = scorer.parse_solver_arm_group(cfg.values, roles['solver'], roles['perception'], cfg.values['knowledge_stores'])
    scorer_id = registered_scorer(cfg.values, bundle.manifest)
    out.mkdir(parents=True, exist_ok=True)
    rendered = scorer.render_supply_variant_first_visit(bundle, group,
        {r['sample_id']: r['stratum'] for r in roster['items']}, out,
        run_id=rid, attempt_id=attempt, roster_sha256=spec['roster']['sha256'], config_hash=cfg.config_hash)
    if rendered['arm_order'] != list(ARMS):
        raise ValueError('exact gepa/lock native arm order required')
    scorer.score_supply_variant_copies(scorer_id, rendered, out, load_lock(umbrella_lock_path(repo)), data,
        scorer.sealed_per_sample(directory, rid, scorer_id))
    results = {}
    for arm in ARMS:
        doc = read(out/f'arm-{arm}-{scorer_id}-scores.json')
        if set(doc) != {scorer_id}:
            raise ValueError('unexpected arm scorer')
        results[arm] = doc[scorer_id]
    verify_delivered_scores(results['lock'], scorer.sealed_per_sample(directory, rid, scorer_id))
    result = summarize(results, roster['items'])
    result.update(schema='saea-gepa-probe-result-v1', new_model_calls=0,
        official_arm_scoring_jobs=2, source=rendered['source'],
        analysis_recipe_sha256=sha(pkg/'analysis-recipe-v1.json'),
        score_receipts={a: pin(data, str((out/f'arm-{a}-{scorer_id}-scores.json.receipt.json').relative_to(data))) for a in ARMS},
        native_role_costs=native_costs(bundle.outputs_rows, bundle.trace_records),
        total_cost=bundle.manifest['cost'])
    path = out/'complete.json'
    if path.exists():
        if path.read_bytes() != raw(result):
            raise ValueError('existing probe analysis differs')
    else:
        publish_new(path, raw(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'data', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--name', required=True)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))
