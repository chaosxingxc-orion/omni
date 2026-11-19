"""Offline node strata for the original sampler; never rewrite historical outcomes."""
import copy
import math
from collections import Counter

from run_package import read, write, pin


METRICS = {'P': 'coverage_at3', 'S1': 'exact', 'S2': 'conditional_on_supply.f1',
           'D': 'joint_exact'}


POSITION = 'position_reward_v1'


def bind_objective(spec, learning):
    result = copy.deepcopy(spec)
    if result['node'] == 'P':
        weights = learning['scorer']['metric_weights']['P']
        if weights not in ({POSITION: 1.0}, {'coverage_at3': 1.0}):
            raise ValueError('unknown P sampling objective')
        metric = next(iter(weights))
        if result.setdefault('metric', metric) != metric:
            raise ValueError('P sampling metric differs from training objective')
    return result


def prepare_selection(sampling, screen_ref, output):
    # The cache contains raw historical node evidence, shared across objectives.
    spec = sampling.get('node_difficulty', {})
    if 'metric' not in spec:
        return _prepare_cached(sampling, screen_ref, output)
    metric = spec['metric']
    if spec.get('node') != 'P' or metric not in (POSITION, METRICS['P']):
        raise ValueError('unknown explicit node sampling metric')
    clean = dict(sampling, node_difficulty={k:v for k,v in spec.items() if k!='metric'})
    prepared = _prepare_cached(clean, screen_ref, output)
    return dict(prepared, node_difficulty=dict(prepared['node_difficulty'], metric=metric))


def position_from_call(part, score):
    calls = part.get('reward_calls')
    if not isinstance(calls, list) or len(calls)!=1 or not calls[0].get('call_ids'):
        raise ValueError('P rank sampling requires original action and gold evidence')
    c = calls[0]
    if c['labels']['intent'] not in c['online']['retrieved_candidates']:
        raise ValueError('P sampling population contains unadmitted gold intent')
    actual = score('P', c['online'], c['output'], c['labels'])
    m = actual['metrics']
    if (actual['mask'] != 1 or bool(actual.get('observed_model_failure')) != bool(part.get('observed_model_failure'))
            or m['coverage_at3'] != part['coverage_at3']
            or (POSITION in part and m[POSITION] != part[POSITION])):
        raise ValueError('closed P call contradicts historical metrics')
    return m[POSITION]


def _prepare_cached(sampling, screen_ref, output):
    """Optionally share one CPU reference preparation across node branches."""
    from pathlib import Path
    spec = sampling.get('node_difficulty')
    if not spec or 'cache_root' not in spec:
        return _prepare_selection(sampling, screen_ref, output)
    if set(spec) != {'node', 'training_builder', 'cache_root'}:
        raise ValueError('shared preparation cache requires an explicit TRAIN builder')
    configured = copy.deepcopy(sampling)
    root = Path(configured['node_difficulty'].pop('cache_root')).resolve()
    if not Path(spec['cache_root']).is_absolute():
        raise ValueError('resolved absolute local reference cache root required')
    from storage import storage
    from training_references import digest
    import fcntl
    storage.check_paths({'node_reference_cache': root}, min_free_fraction=0.10)
    identity = dict(screen_ref=pin(screen_ref),
        training_builder={k:pin(v) for k,v in spec['training_builder'].items()})
    entry = root/digest(identity)
    entry.mkdir(parents=True, exist_ok=True)
    with (entry/'PREPARE.guard').open('ab') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        return _prepare_selection(configured, screen_ref, entry)


def _prepare_selection(sampling, screen_ref, output):
    """Resolve configured contribution reports once in the existing CPU prepare.

    A durable receipt permits restart without joining the full population again.
    Only the new dataset's sampling configuration receives the derived reference.
    """
    from pathlib import Path
    spec = sampling.get('node_difficulty')
    if spec is None:
        return sampling
    if spec.get('node') not in METRICS:
        raise ValueError('known node difficulty required')
    if set(spec) == {'node', 'scores_ref'}:
        return sampling
    root = Path(output); marker = root/'PREPARED.json'
    builder_identity = None
    if set(spec) == {'node', 'training_builder'}:
        # A completed score document contains all four nodes. Recovery should
        # not launch one interpreter per historical batch just to rediscover it.
        binding = spec['training_builder']
        if set(binding) != {'module_ref', 'metrics_pin_ref'}:
            raise ValueError('explicit TRAIN contribution builder and metrics required')
        builder_identity = dict(screen_ref=pin(screen_ref),
            training_builder={k: pin(v) for k,v in binding.items()})
        if marker.exists():
            receipt = read(marker)
            if 'builder_identity' in receipt:
                if receipt['builder_identity'] != builder_identity:
                    raise ValueError('TRAIN builder identity changed during recovery')
                result = copy.deepcopy(sampling)
                result['node_difficulty'] = dict(node=spec['node'], scores_ref=pin(receipt['scores_ref']))
                return result
        spec = dict(node=spec['node'], contribution_refs=prepare_training_contributions(
            screen_ref, spec['training_builder'], output))
    if set(spec) != {'node', 'contribution_refs'} or not spec['contribution_refs']:
        raise ValueError('configure scores_ref or contribution_refs, never both')
    identity = dict(screen_ref=pin(screen_ref),
                    contribution_refs=[pin(r) for r in spec['contribution_refs']])
    if marker.exists():
        receipt = read(marker)
        if receipt['identity'] != identity:
            raise ValueError('node score preparation input changed during recovery')
        score_ref = pin(receipt['scores_ref'])
    else:
        score_ref = prepare_scores(identity['screen_ref'], identity['contribution_refs'], root/'SCORES.json')
        receipt = dict(schema='prepared-node-difficulty-v1', identity=identity,
                       scores_ref=score_ref, model_calls=0)
        if builder_identity is not None:
            receipt['builder_identity'] = builder_identity
        write(marker, receipt)
    result = copy.deepcopy(sampling)
    result['node_difficulty'] = dict(node=spec['node'], scores_ref=score_ref)
    return result


def prepare_training_contributions(screen_ref, binding, output):
    """Produce node references from the union's closed TRAIN batches on CPU.

    Reuse the existing DATA scorer in its own interpreter. This is not a second
    controller and never contacts a model. Completed batches resume in-place.
    """
    import subprocess
    import sys
    from pathlib import Path
    from dependencies import local_path
    if set(binding) != {'module_ref', 'metrics_pin_ref'}:
        raise ValueError('explicit TRAIN contribution builder and metrics required')
    module = pin(binding['module_ref']); metrics = pin(binding['metrics_pin_ref'])
    if Path(module['path']).name != 'node_contributions.py':
        raise ValueError('original node contribution builder required')
    screen = read(screen_ref)
    if not screen.get('sealed') or screen.get('synthetic_fixture') or not screen.get('runs'):
        raise ValueError('closed real TRAIN run union required')
    root = Path(output)/'training-contributions'; refs=[]; seen=set()
    for run in screen['runs']:
        seal_ref = pin(run['complete'])
        if seal_ref['sha256'] in seen:
            raise ValueError('duplicate closed TRAIN batch')
        seen.add(seal_ref['sha256'])
        seal = read(seal_ref); version = read(seal['version'])
        if seal['version'] != run['version'] or version['system_version'] != run['runtime_system']:
            raise ValueError('TRAIN contribution run identity differs')
        destination = root/seal_ref['sha256']
        subprocess.run([sys.executable, '-B', str(local_path(module['path'])),
            '--training-seal', str(local_path(seal_ref['path'])),
            '--metrics-pin', str(local_path(metrics['path'])),
            '--destination', str(destination)], check=True, stdout=subprocess.DEVNULL)
        refs.append(pin(destination/'NODE-CONTRIBUTIONS.json'))
    return refs


def prepare_scores(screen_ref, contribution_refs, output):
    """Join existing offline node_contributions, without model calls or rescoring."""
    from cohort_io import imports, BASE
    manifest = read(screen_ref)
    with imports([BASE/'collection-scale-successor/code'], ('common', 'stream_pool')):
        from stream_pool import Pool, iter_jsonl
        pool = Pool(manifest) if 'streams' in manifest else manifest
        wanted = {r['recording_ref_id']: (r['sample_id'], r['source_id'], r['system_version']) for r in pool['records']}
        rows = {}; sources = []
        for ref in contribution_refs:
            report = read(ref)
            if report.get('schema') != 'offline-node-contributions-v2':
                raise ValueError('original node contribution report required')
            evaluation = read(report['report'])
            if evaluation.get('split') != 'TRAIN':
                raise ValueError('heldout node contributions cannot select training samples')
            sources.append(pin(ref))
            for item in iter_jsonl(report['records']):
                rid = item['reference_id']
                if rid not in wanted:
                    continue
                if rid in rows or wanted[rid] != (item['sample_id'], item['source_id'], evaluation['system']['sha256']):
                    raise ValueError('duplicate/foreign node contribution identity')
                rows[rid] = dict(schema=item['node_metric_schema'], reference_id=rid,
                    sample_id=item['sample_id'], source_id=item['source_id'],
                    node_statistics=item['node_statistics'], evidence=item['node_evidence'],
                    annotation_sha256=item['node_annotation_sha256'], gold_evidence=item['node_gold_evidence'],
                    offline_gold_only=True)
        if set(rows) != set(wanted):
            raise ValueError('incomplete admitted node contribution coverage')
    return write(output, dict(schema='frozen-node-difficulty-v1', screen_version=manifest['version'],
        system_version=manifest.get('scientific_identity', manifest.get('system_version')),
        screen_ref=pin(screen_ref), contribution_refs=sources, split='TRAIN', offline_gold_only=True,
        records=[rows[k] for k in sorted(rows)], model_calls=0))


def projection(records, manifest, specification):
    """Consume pinned, complete offline node scores after original pool validation.

    The score producer is the existing node_metrics assessor. Missing scores are
    not errors or successes. S2 without any actual supplied type has no action;
    count it separately rather than spend rollout requests on an empty task.
    """
    if set(specification) not in ({'node','scores_ref'}, {'node','scores_ref','metric'}) or specification['node'] not in METRICS:
        raise ValueError('explicit node and pinned offline scores required')
    node = specification['node']; ref = specification['scores_ref']
    if not isinstance(ref, dict) or not ref.get('sha256'):
        raise ValueError('node difficulty scores must be pinned')
    metric = specification.get('metric', METRICS[node])
    if metric != METRICS[node] and not (node=='P' and metric==POSITION):
        raise ValueError('invalid difficulty metric for node')
    if metric == POSITION:
        from cohort_io import imports, BASE, load
        with imports([BASE/'unified-learning'], ('kernel',)):
            score = load(BASE/'unified-learning/scoring.py','offline_sampling_position').score_component
    scores = read(ref)
    if (scores.get('schema') != 'frozen-node-difficulty-v1' or
            scores.get('screen_version') != manifest['version'] or
            scores.get('system_version') != manifest.get('scientific_identity', manifest.get('system_version')) or
            scores.get('split') != 'TRAIN' or scores.get('offline_gold_only') is not True):
        raise ValueError('node difficulty population/system identity differs')
    index = {}
    for row in scores['records']:
        rid = row['reference_id']
        if rid in index:
            raise ValueError('duplicate node difficulty recording')
        index[rid] = row
    if set(index) != {r['recording_ref_id'] for r in records}:
        raise ValueError('node difficulty must account for every admitted recording')
    original = {}; projected = []; excluded = Counter()
    for record in records:
        row = index[record['recording_ref_id']]
        if (row['sample_id'] != record['sample_id'] or
                row.get('source_id') != record['source_id'] or
                row.get('schema') != 'actual-node-statistics-v2' or
                row.get('offline_gold_only') is not True or
                not row.get('annotation_sha256') or not row.get('gold_evidence') or
                not row.get('evidence', {}).get('outcome_sha256')):
            raise ValueError('node difficulty lacks observed input/gold provenance')
        part = row['node_statistics'][node]
        if node == 'S2' and part.get('no_branch_path') == 1:
            excluded['no_supplied_S2_action'] += 1
            continue
        observed = part.get('conditional_on_supply', {}) if node == 'S2' else part
        if observed.get('complete') is not True or observed.get('mask') != 1:
            raise ValueError('incomplete node difficulty: ' + record['recording_ref_id'])
        if metric == POSITION:
            value = position_from_call(part, score)
        else:
            value = part
            for key in metric.split('.'):
                value = value[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid node difficulty metric')
        sid = record['sample_id']; original[sid] = record
        # This view exists only inside the original stratified sampler. Restore
        # original records before publishing draws/references or current rollout.
        projected.append(dict(record, outcome='success' if value == 1 else 'error'))
    return projected, original, dict(excluded)


def restore(draws, statistics, originals, specification, excluded):
    result = []
    for draw in draws:
        item = dict(draw)
        item['record'] = originals[draw['record']['sample_id']]
        item['difficulty'] = dict(node=specification['node'], metric=specification.get('metric', METRICS[specification['node']]),
                                  outcome=draw['cell']['outcome'], scores_ref=specification['scores_ref'])
        result.append(item)
    stats = copy.deepcopy(statistics)
    stats.update(difficulty_node=specification['node'], difficulty_metric=specification.get('metric', METRICS[specification['node']]),
                 node_outcomes=stats['outcomes'],
                 selected_system_outcomes=dict(Counter(d['record']['outcome'] for d in result)),
                 node_exclusions=excluded, original_outcomes_preserved=True)
    return result, stats


if __name__ == '__main__':
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args(); config = read(a.config)
    from fresh_adapter import scope
    with scope():
        result = prepare_scores(config['screen_ref'], config['contribution_refs'], config['output'])
    print(json.dumps(result))
