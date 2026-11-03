"""Read finalized fixed development arms, official paired scores and native costs.

No model or scorer calls. Output contains aggregates and provenance, no raw text.
"""
import argparse
import base64
import collections
import copy
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
import tfrl_h1_observation_read as observation
from tfrl_h1_observation import (identity, read, raw, sha, pinned,
    comparison_config, population_scope, frozen_candidate)
from tfrl_current_half_result import score_table
from speech_aware_evidence_acquisition.analysis.paired import resampled_statistic
from paired_ci import pooled_slu_f1
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

ARMS = ('off', 'short', 'baseline')
SEED = 20261103
DRAWS = 10000


def config(arm, half='h1'):
    return comparison_config({'schema':f'saea-{half}-perception-thinking-v1',
        'arm':arm, 'admitted_level':96})


def comparison_scope(half):
    _, _, rows = population_scope(config('off', half))
    return (ARMS if half == 'h1' else ('off', 'baseline')), rows


def read_scope(spec):
    half = {'saea-h1-thinking-comparison-read-v1':'h1',
        'saea-h2-thinking-comparison-read-v1':'h2'}.get(spec.get('schema'))
    if half is None:
        raise ValueError('explicit H1 or H2 comparison schema required')
    arms, rows = comparison_scope(half)
    keys = {'schema', 'data', 'arms'}
    if half == 'h2':
        keys.add('candidate_freeze')
    if set(spec) != keys or not isinstance(spec['arms'], dict) or set(spec['arms']) != set(arms):
        raise ValueError('exact arms and freeze for declared comparison required')
    return half, arms, rows


def assert_candidate_freeze(repo, comparison, candidate, expected):
    actual = frozen_candidate(repo, comparison)
    if actual != expected or candidate.get('candidate_freeze') != actual:
        raise ValueError('H2 comparison differs from preregistered candidate freeze')
    return actual


def invariant(base):
    value = copy.deepcopy(base)
    for key in ('thinking_policy', 'prepared_phase_supply', 'what_this_is'):
        value.pop(key, None)
    for key in ('run_id', 'attempt_id'):
        value['native_contact_fit'].pop(key, None)
    return value


def paired_scores(results, items, *, draws=DRAWS, half='h1'):
    arms, rows = comparison_scope(half)
    if set(results) != set(arms):
        raise ValueError('all declared complete thinking arms required')
    ids = [row['sample_id'] for row in items]
    if len(ids) != rows or len(set(ids)) != rows:
        raise ValueError(f'exact full {half.upper()} {rows} required')
    tables = {arm:score_table(value, ids) for arm,value in results.items()}
    keys = ('evaluator_commit','scorer_version','split_id','projection','filler_normalization')
    official = lambda value: {k:value['aggregate'][k] for k in keys}
    if any(official(v) != official(results['baseline']) for v in results.values()):
        raise ValueError('official evaluator identities differ')
    counts = lambda r: (r['slot_f1_exact_fp'], r['slot_f1_exact_fn'])
    success = {}
    for arm,table in tables.items():
        for row in table.values():
            if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 or int(v) != v
                   for v in counts(row)):
                raise ValueError('finite nonnegative exact slot counts required')
        success[arm] = {sid:int(table[sid]['scenario_action_exact'] == 1 and counts(table[sid]) == (0,0)) for sid in ids}
    out = {}
    for arm in arms:
        if arm == 'baseline':
            continue
        left,right = tables['baseline'],tables[arm]
        stat = {
            'slu_f1_pp':lambda sample:100*(pooled_slu_f1(right,sample)-pooled_slu_f1(left,sample)),
            'intent_pp':lambda sample:100*sum(right[s]['scenario_action_exact']-left[s]['scenario_action_exact'] for s in sample)/len(sample)}
        contrasts = {metric:resampled_statistic(sample_ids=ids,statistic=fn,seed=SEED,
            n_replicates=draws,ci_level=.95,cluster_of=lambda s:':'.join(s.split(':')[:3])).as_json()
            for metric,fn in stat.items()}
        saves = sum(success[arm][s] > success['baseline'][s] for s in ids)
        harms = sum(success[arm][s] < success['baseline'][s] for s in ids)
        out[arm] = {'minus':'baseline', 'contrasts':contrasts,
            'task_repairs':saves,'task_breaks':harms,'task_ties':rows-saves-harms,
            'task_success_rule':'Intent exact AND exact-slot FP=FN=0'}
    return {'rows':rows,'official':{a:v['aggregate']['official'] for a,v in results.items()},
        'comparisons':out,'bootstrap_unit':'utterance cluster', 'seed':SEED,
        'decoding_noise_estimated':False,'automatic_adoption':False,
        'limitations':['Fixed hard development set, not natural-distribution test performance.',
            'Paired intervals describe roster uncertainty, not independent decoding noise.',
            'Intervals are marginal exploratory 95% intervals; no multiple-comparison correction.',
            'An interval crossing zero does not establish equivalence.']}


def distribution(values):
    values = sorted(values)
    return {'n':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),
        'p95':values[int(.95*(len(values)-1))]} if values else None


def native_costs(outputs, trace):
    """Attribute every sealed COST record; retain extra/fallback contacts explicitly."""
    roles = {}
    for row in outputs:
        for visit in row['native_episode_receipt']['role_visits']:
            for contact in visit['execution_receipt']['contacts']:
                rid = contact['request_id']
                if rid in roles:
                    raise ValueError('duplicate request attribution')
                roles[rid] = visit['role_id']
    groups = collections.defaultdict(list)
    seen = set()
    for key,value in trace.items():
        if not key.endswith('/cost') or 'calls' not in value:
            continue
        rid = key.removesuffix('/cost'); seen.add(rid)
        for field in ('calls','prompt_tokens','completion_tokens','latency_seconds'):
            v=value[field]
            if type(v) not in (int,float) or not math.isfinite(v) or v < 0:
                raise ValueError('invalid native cost')
        wire = trace.get(rid+'/wire-response')
        detail = {'wire_present':wire is not None}
        if wire is not None:
            document = json.loads(base64.b64decode(wire['response_b64'],validate=True))
            choice, = document['choices'];msg=choice['message']
            reasoning=msg.get('reasoning_content')
            stats=msg.get('reasoning')
            chars=(stats.get('reasoning_chars',0) if isinstance(stats,dict)
                   else len(reasoning) if isinstance(reasoning,str)
                   else len(stats) if isinstance(stats,str) else 0)
            content=msg.get('content') or ''
            detail.update(reasoning_chars=chars,think_marker='<think>' in content,
                finish_reason=choice.get('finish_reason'),empty_content=not content.strip())
        groups[roles.get(rid,'unattributed')].append({**value,**detail})
    if set(roles)-seen:
        raise ValueError('role contacts lack native cost records')
    result = {}
    for role,rows in groups.items():
        result[role] = {field:sum(r[field] for r in rows) for field in ('calls','prompt_tokens','completion_tokens')}
        result[role].update(request_seconds=distribution([r['latency_seconds'] for r in rows]),
            wire_records=sum(r['wire_present'] for r in rows),
            contacts_with_reasoning=sum(r.get('reasoning_chars',0)>0 for r in rows),
            think_markers=sum(r.get('think_marker',False) for r in rows),
            empty_content=sum(r.get('empty_content',False) for r in rows),
            finish_reasons=dict(collections.Counter(r['finish_reason'] for r in rows if r['wire_present'])))
    return result



def phase_timing(directory, run_id, timing_path, expected_ids, checkpoint_sha256):
    """Join the exact sealed arm checkpoint; expose incomplete timing explicitly."""
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from real_engine_phase_timing import analyze
    checkpoint = directory / (run_id + '.rows.ckpt.jsonl')
    rows = read_checkpoint(checkpoint, expected_sha256=checkpoint_sha256).rows
    actual = [row.sample_id for row in rows if row.keep_defect() is None]
    if len(actual) != len(expected_ids) or set(actual) != set(expected_ids):
        raise ValueError('phase timing checkpoint differs from finalized arm roster')
    before = sha(timing_path)
    result = analyze(checkpoint, timing_path)
    if sha(timing_path) != before:
        raise ValueError('timing input changed during finalized analysis')
    result['timing_sha256'] = before
    result['complete_request_timing'] = (
        result['rows'] == len(expected_ids) and result['missing_timing_entries'] == 0
        and not result['timing_partial_tail'])
    return result


def run(spec_path, out):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    spec=read(spec_path)
    half, arms, rows = read_scope(spec)
    data=Path(spec['data']);result={};scores={};invariants=[];bindings=[];roster=None
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('owned CPU output required')
    for arm in arms:
        entry=spec['arms'][arm]
        if set(entry) not in ({'repo','attempt_id'}, {'repo','attempt_id','timing'}):
            raise ValueError('explicit repository and attempt per arm required')
        repo=Path(entry['repo']);comparison=config(arm, half)
        name,rid=identity(1,comparison);attempt=entry['attempt_id']
        pkg=repo/'docs/readiness'/name;candidate=read(pkg/'candidate.json')
        if half == 'h2':
            assert_candidate_freeze(repo, comparison, candidate, spec['candidate_freeze'])
        result[arm]=observation.run(repo,data,out/arm,comparison=comparison,attempt_id=attempt)
        bindings.append({k:candidate[k] for k in ('roster','dataset','runtime_binding','gain_lock')})
        invariants.append(invariant(read(repo/f'configs/baseline/{name}.json')))
        roster=pinned(repo,candidate['roster'])
        directory=data/'runs'/rid/attempt;bundle=verify_bundle(directory/(rid+'.run-manifest.json'))
        scores[arm]=read(directory/(rid+'.scores.json'))['slurp-slu-official-train']
        result[arm]['native_role_costs']=native_costs(bundle.outputs_rows,bundle.trace_records)
        if 'timing' in entry:
            result[arm]['phase_timing'] = phase_timing(directory, rid, Path(entry['timing']),
                [item['sample_id'] for item in roster['items']], bundle.manifest['row_checkpoint']['sha256'])
    if any(x != bindings[0] for x in bindings) or any(x != invariants[0] for x in invariants):
        raise ValueError('comparison changed fixed data, runtime, knowledge or non-thinking policy')
    summary={'schema':f'saea-{half}-thinking-comparison-result-v1', 'arms':result,
        'paired':paired_scores(scores,roster['items'],half=half), 'source_config_sha256':sha(spec_path),
        'new_model_calls':0,'new_scoring_jobs':0,'test_used_for_tuning':False,
        'request_latency_is_not_wall_throughput':True}
    if half == 'h2':
        summary['candidate_freeze'] = spec['candidate_freeze']
        summary['paired']['limitations'].append(
            'H2 has prior development-selection exposure; it is not an untouched test set.')
    path=out/'complete.json'
    if path.exists():
        if path.read_bytes()!=raw(summary):raise ValueError('existing comparison differs')
    else:publish_new(path,raw(summary))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.config,args.out),sort_keys=True))
