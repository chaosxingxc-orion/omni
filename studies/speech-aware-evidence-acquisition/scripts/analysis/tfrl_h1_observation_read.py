"""Read the finalized full H1 lock observation; never rescore or call a model."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from tfrl_h1_observation import identity, read, raw, sha, pinned
from tfrl_current_half_result import score_table
from speech_aware_evidence_acquisition.core.atomic_file import publish_new


def summarize_scores(result, items):
    ids = [r['sample_id'] for r in items]
    if len(ids) != 313 or len(set(ids)) != 313:
        raise ValueError('exact full H1 required')
    table = score_table(result, ids)
    correct = []
    for sid in ids:
        r = table[sid]
        counts = [r[k] for k in ('slot_f1_exact_fp', 'slot_f1_exact_fn')]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 or x != int(x) for x in counts):
            raise ValueError('finite nonnegative exact slot counts required')
        correct.append(int(r['scenario_action_exact'] == 1 and counts == [0, 0]))
    return {'rows': 313, 'official': result['aggregate']['official'],
            'intent_accuracy': sum(table[s]['scenario_action_exact'] for s in ids)/313,
            'task_correct_rows': sum(correct), 'task_error_rows': 313-sum(correct),
            'task_success_rule': 'Intent exact AND exact-slot FP=FN=0',
            'hardness_confirmed': False, 'independent_stochastic_draws_claimed': False}


def role_costs(outputs):
    """Receipt-based request attribution, including any additional contacts."""
    roles = collections.defaultdict(collections.Counter)
    seen = set()
    for row in outputs:
        for visit in row['native_episode_receipt']['role_visits']:
            for contact in visit['execution_receipt']['contacts']:
                rid = contact['request_id']
                if rid in seen:
                    raise ValueError('duplicate role request attribution')
                seen.add(rid)
                roles[visit['role_id']]['calls'] += 1
    return {k: dict(v) for k,v in roles.items()}


def run(repo, data, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from registered_workload import validate_workload
    assert_execution_environment()
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('owned CPU result directory required')
    name, rid = identity(1)
    pkg = repo/'docs/readiness'/name
    spec = read(pkg/'candidate.json')
    validate_workload(repo,data,name,spec)
    recipe = read(pkg/'analysis-recipe-v1.json')
    if (recipe['schema'] != 'saea-h1-observation-read-v1' or recipe['rows'] != 313
            or recipe['generation_run_id'] != rid or recipe['attempt_id'] != 'generation-v1'
            or recipe['round'] != 1 or recipe['roster'] != spec['roster']
            or recipe['new_model_calls'] != 0 or recipe['claims_benefit'] is not False):
        raise ValueError('exact preregistered full H1 analysis required')
    directory = data/'runs'/rid/'generation-v1'
    mp = directory/(rid+'.run-manifest.json')
    bundle = verify_bundle(mp)
    roster = pinned(repo,spec['roster'])
    ids = [r['sample_id'] for r in roster['items']]
    if (bundle.attempt_state != 'FINALIZED' or bundle.manifest['status'] != 'COMPLETED'
            or bundle.manifest['run_id'] != rid
            or bundle.manifest['attempt']['attempt_id'] != 'generation-v1'
            or len(bundle.split_ids) != 313 or set(bundle.split_ids) != set(ids)):
        raise ValueError('finalized complete313 required')
    f = bundle.manifest['config']['fragments']
    cfg = compose(repo, *(f[k] for k in ('model','dataset','baseline','experiment')))
    if cfg.config_hash != bundle.manifest['config']['config_hash']:
        raise ValueError('sealed configuration changed')
    scorer = 'slurp-slu-official-train'
    if cfg.values['scorers'] != [scorer,'cost'] or bundle.manifest['scorers'] != [scorer,'cost']:
        raise ValueError('frozen raw official TRAIN scorer required')
    scores_path = directory/(rid+'.scores.json')
    scores = read(scores_path)
    if set(scores) != {scorer,'cost'}:
        raise ValueError('unexpected sealed scorers')
    result = summarize_scores(scores[scorer],roster['items'])
    result.update(schema='saea-h1-fresh-observation-result-v1',run_id=rid,attempt_id='generation-v1',
        scorer_id=scorer,role_contacts=role_costs(bundle.outputs_rows),cost=bundle.manifest['cost'],
        new_model_calls=0,new_scoring_jobs=0,claims_benefit=False,
        historical_costs_excluded_from_this_run=True,
        source={'manifest_sha256':sha(mp),'scores_sha256':sha(scores_path),
                'roster_sha256':spec['roster']['sha256'],'config_hash':cfg.config_hash,
                'recipe_sha256':sha(pkg/'analysis-recipe-v1.json')},
        limitations=['One fresh lock observation does not establish long-run error probability.',
                     'Historical failed runs and learning costs remain in their original ledgers.',
                     'No comparison with a different roster or shared-P K4 as an accuracy gain.'])
    out.mkdir(parents=True,exist_ok=True)
    path=out/'complete.json'
    if path.exists():
        if path.read_bytes()!=raw(result):raise ValueError('existing result differs')
    else:publish_new(path,raw(result))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','out'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),sort_keys=True))
