"""One deterministic sampler/publisher, whole-query or whole-group sampling only."""
import collections
import importlib.util
import json
import math
from pathlib import Path
import random
from kernel import require, digest, parent, runtime, load

POPULATION_MODULE = Path(__file__).resolve().parents[1] / 'collection-scale-successor/code/train_population.py'


def pinned(ref, base):
    path = Path(ref['path'])
    if not path.is_absolute():
        path = Path(base) / path
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
            (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'input changed while reading')
    import hashlib
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'] and ref.get('bytes',len(raw)) == len(raw), 'pin mismatch: '+str(path))
    return raw


def quota(count, weights):
    require(type(count) is int and count >= 0 and weights, 'invalid count/weights')
    require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in weights.values()) and abs(math.fsum(weights.values())-1) < 1e-10, 'mix must sum to one')
    values = {k:count*w for k,w in weights.items()}
    result = {k:math.floor(v) for k,v in values.items()}
    for k in sorted(values,key=lambda k:(-(values[k]-result[k]),k))[:count-sum(result.values())]:
        result[k] += 1
    return result


def check_pool(pool,c,base='.'):
    require(pool['schema'] == 'sealed-training-pool-v1' and pool['sealed'] is True, 'unsealed pool')
    require(pool['version'] == c['pool_version'] and pool['system_version'] == c['system_version'], 'pool/system version mismatch')
    require(pool['lineage'] and all(pool['lineage'].get(n) for n in ('raw_dataset','system_release','knowledge','retrieval','prompt','rails','scorer')), 'incomplete lineage')
    require(pool['official_train_scope'] == 'entire_official_TRAIN_real_and_synthetic' and pool['population_complete'] is True, 'incomplete pool; no ready dataset')
    require(type(pool['synthetic_fixture']) is bool, 'explicit synthetic pool provenance')
    if not pool['synthetic_fixture']:
        ref = pool['lineage']['raw_dataset']
        require(isinstance(ref,dict) and ref.get('sha256'), 'official dataset pin required')
        expected = load('official_train_population',POPULATION_MODULE).official_domains(json.loads(pinned(ref,base)))
        require(collections.Counter(r['domain'] for r in pool['population']) == expected, 'official TRAIN domain denominator mismatch')
    roster = pool['population']
    ids = [r['recording_ref_id'] for r in roster]
    require(len(set(ids)) == len(ids) and len(ids) == pool['official_train_count'] and set(r['domain'] for r in roster) == {'real','synthetic'}, 'complete official TRAIN roster required')
    require(all(r['split'] == 'TRAIN' and r['domain'] in ('real','synthetic') for r in roster), 'TEST/DEV excluded from TRAIN population')
    require(pool['population_sha256'] == digest(roster) and pool['cross_project_exclusions'] is False, 'population provenance mismatch')
    outcomes = pool['outcomes']
    require(len(outcomes) == len(ids) and {o['recording_ref_id'] for o in outcomes} == set(ids), 'unaccounted population outcomes')
    by_source = {r['recording_ref_id']:r for r in roster}
    by_outcome = {}
    for o in outcomes:
        require(o['status'] in ('error','success','unknown','ineligible') and type(o['gold_in_actual_pre_P_dynamic_retrieval']) is bool, 'unknown outcome category')
        require(o['status'] not in ('error','success') or o['gold_in_actual_pre_P_dynamic_retrieval'], 'admitted outcome requires pre-P coverage; never gate on P@3')
        by_outcome[o['recording_ref_id']] = o
    records, seen = [],set()
    excluded = collections.Counter()
    for r in pool['records']:
        require(r['sample_id'] not in seen, 'duplicate sample record')
        seen.add(r['sample_id'])
        src = by_source.get(r['recording_ref_id'])
        require(src is not None and r['split'] == 'TRAIN' and r['domain'] == src['domain'] and r['source'] == src['source'] and r['source_id'] == src['source_id'], 'TEST/unknown source/split/domain mismatch')
        o = by_outcome[r['recording_ref_id']]
        if o['status'] not in ('error','success') or not o['gold_in_actual_pre_P_dynamic_retrieval']:
            excluded[o['status']] += 1
            continue
        require(r['outcome'] == o['status'], 'record outcome mismatch')
        require(r['system_version'] == c['system_version'] and r['rollout_scope'] == c['scope'], 'scope/system mismatch')
        require(r.get('online_gold_injected') is False and 'online' in r and 'offline_supervision' in r, 'separate actual online/offline data required')
        records.append(r)
    represented = {r['recording_ref_id'] for r in records}
    admitted = {o['recording_ref_id'] for o in outcomes if o['status'] in ('error','success')}
    require(represented == admitted, 'incomplete admitted trajectory export')
    return sorted(records,key=lambda r:r['sample_id']),dict(excluded),by_outcome


def tensor_shape(value):
    if not isinstance(value,list):
        require(type(value) in (int,float) and math.isfinite(value), 'invalid tensor scalar')
        return []
    if not value:return [0]
    shapes=[tensor_shape(v) for v in value]
    require(all(s == shapes[0] for s in shapes), 'ragged captured tensor')
    return [len(value)]+shapes[0]


def tensor_values(value):
    if isinstance(value,list):
        for v in value:yield from tensor_values(v)
    else:yield value


def fresh_group_check(record,c,base):
    require(record.get('purpose') == 'fresh_on_policy', 'historical trajectory is not a fresh group')
    b = record.get('behavior_bundle')
    require(isinstance(b,dict), 'missing complete fresh behavior group')
    from schedule import Clock
    from learning import evaluate
    learning_config = json.loads(pinned(c['learning_config_ref'],base))
    require(learning_config['system_version'] == c['system_version'], 'learning system mismatch')
    clock = Clock(learning_config,record['clock_state'])
    require(len(b['groups']) == 1 and b['groups'][0]['snapshot']['rollout_scope'] == c['scope'], 'sample must be exactly one complete scope group')
    evaluate(b,clock)
    require(b['synthetic'] == record['synthetic_fixture'], 'synthetic group provenance mismatch')
    for e in b['groups'][0]['episodes']:
        for call in e['calls']:
            fields = call.get('training_fields',{})
            require(all(fields.get(f) is not None for f in parent.FIELDS), 'missing model-training modality/token/probability fields')
            require(isinstance(fields['prompt_token_ids'],list) and fields['prompt_token_ids'] and
                    all(type(v) is int and v >= 0 for v in fields['prompt_token_ids']), 'missing actual prompt token sequence')
            length=len(fields['prompt_token_ids'])+len(fields['action_token_ids'])
            require(isinstance(fields['attention_mask'],list) and len(fields['attention_mask']) == length and
                    all(type(v) is int and v in (0,1) for v in fields['attention_mask']), 'invalid full attention mask')
            require(fields['action_mask'] == call['action_mask'] and fields['behavior_policy_version'] == call['behavior_policy_version']
                    and fields['adapter_vector'] == call['adapter_vector'] and fields['probability_semantics'] == call['probability_semantics'], 'modality/policy evidence mismatch')
            schema=fields.get('captured_tensor_schema')
            require(isinstance(schema,dict) and schema.get('processor_manifest_sha256') == digest(fields['processor_manifest'])
                    and schema.get('consumer_contract_version'), 'captured processor/consumer tensor contract required')
            for key in ('audio_features','feature_mask','position_ids','effective_audio_lengths'):
                spec=schema['tensors'][key]
                actual_shape=tensor_shape(fields[key])
                require(actual_shape == spec['shape'] and len(spec['axes']) == len(actual_shape)
                        and len(set(spec['axes'])) == len(spec['axes']), 'captured tensor shape/axes mismatch: '+key)
                for coordinate,axis in enumerate(spec['axes']):
                    if axis == 'tokens':require(actual_shape[coordinate] == length, 'processor token axis mismatch')
            features=schema['tensors']['audio_features'];mask=schema['tensors']['feature_mask']
            require(all(axis in features['axes'] for axis in mask['axes']), 'feature mask axes outside features')
            require(all(features['shape'][features['axes'].index(axis)] == mask['shape'][i] for i,axis in enumerate(mask['axes'])), 'mask/feature dimensions mismatch')
            require(all(type(x) is int and x in (0,1) for x in tensor_values(fields['feature_mask'])), 'invalid feature mask')
            require(all(type(x) is int and x>=0 for x in tensor_values(fields['position_ids'])), 'invalid processor positions')
            require(sum(tensor_values(fields['effective_audio_lengths'])) == sum(tensor_values(fields['feature_mask'])), 'effective audio length/mask mismatch')
            require(all(isinstance(fields[k],dict) and fields[k] for k in ('processor_manifest','tokenizer_manifest','sampling_distribution')), 'unpinned modality/tokenizer/sampling metadata')
            require(fields['action_token_ids'] == [t for t,m in zip(call['token_ids'],call['action_mask']) if m]
                    and fields['behavior_logprobs'] == [p for p,m in zip(call['old_logprobs'],call['action_mask']) if m], 'loader action/behavior fields inconsistent')


def sample(pool,c,base='.'):
    require(c['schema'] == 'dataset-sampling-config-v1' and c['split'] == 'TRAIN', 'TRAIN-only sampling config')
    require(c['purpose'] in ('screen','fresh_on_policy') and c['scope'] in ('node','system'), 'purpose/scope invalid')
    require(type(c['seed']) is int and type(c['sample_count']) is int and c['sample_count'] > 0, 'seed/count invalid')
    require(c['replacement'] in ('forbid','allow') and c['insufficient'] == 'fail', 'explicit replacement or fail required; never silently change ratio')
    require(c['format'] == 'separated-json-v1', 'unsupported training format')
    records,excluded,outcomes = check_pool(pool,c,base)
    if c['purpose'] == 'fresh_on_policy':
        require(c['error_success'] is None, 'on-policy groups cannot be reweighted by rollout success/error')
        for r in records:
            require(r.get('synthetic_fixture') == pool['synthetic_fixture'], 'record/pool synthetic provenance mismatch')
            fresh_group_check(r,c,base)
        outcome_quota = {'all':c['sample_count']}
    else:
        require(set(c['error_success']) == {'error','success'}, 'explicit screen mix required')
        outcome_quota = quota(c['sample_count'],c['error_success'])
    require(set(c['domain_mix']) == {'real','synthetic'}, 'real/synthetic mix must be explicit')
    rng = random.Random(c['seed'])
    draws, cells = [],[]
    for outcome,n in sorted(outcome_quota.items()):
        for domain,dn in sorted(quota(n,c['domain_mix']).items()):
            candidates = [r for r in records if (outcome == 'all' or r['outcome'] == outcome) and r['domain'] == domain]
            balance_key = c['source_balance']['key']
            require(balance_key in ('source','source_id'), 'source balance key must be explicit origin source or text source_id')
            sources = sorted(set(r[balance_key] for r in candidates))
            if dn == 0:
                continue
            require(sources, f'insufficient pool: {outcome}/{domain}')
            balance = c['source_balance']
            if balance['mode'] == 'uniform':
                weights = {s:1/len(sources) for s in sources}
            elif balance['mode'] == 'proportional':
                weights = {s:sum(r[balance_key]==s for r in candidates)/len(candidates) for s in sources}
            else:
                require(balance['mode'] == 'explicit', 'unknown source balance')
                weights = balance['weights'][domain]
                require(set(weights) == set(sources), 'source weight roster mismatch')
            quota(0,weights)  # Validate normalized finite nonnegative source probabilities.
            require(balance['draw_policy'] == 'seeded_source_then_recording', 'explicit stochastic source design required')
            cells_by_source={source:[r for r in candidates if r[balance_key]==source] for source in sources}
            remaining={source:list(cell) for source,cell in cells_by_source.items()}
            if c['replacement']=='forbid':
                require(balance['without_replacement']=='renormalize_nonempty_sources', 'explicit source exhaustion policy required')
                require(dn <= sum(len(cell) for source,cell in remaining.items() if weights[source]>0), 'insufficient pool for positive-weight sources')
            cells.append(dict(outcome=outcome,domain=domain,quota=dn,population=len(candidates),source_probabilities=weights,
                              source_populations={s:len(cell) for s,cell in cells_by_source.items()},balance_key=balance_key))
            for j in range(dn):
                available=[source for source in sources if weights[source]>0 and (c['replacement']=='allow' or remaining[source])]
                total=math.fsum(weights[source] for source in available)
                draw=rng.random()*total
                chosen=available[-1]
                for source in available:
                    draw-=weights[source]
                    if draw<0:
                        chosen=source
                        break
                options=cells_by_source[chosen] if c['replacement']=='allow' else remaining[chosen]
                count=len(options);pick=rng.randrange(count)
                record=options[pick] if c['replacement']=='allow' else options.pop(pick)
                source_probability=weights[chosen]/total
                record_probability=source_probability/count
                inclusion=1-(1-weights[chosen]/len(cells_by_source[chosen]))**dn if c['replacement']=='allow' else None
                draws.append(dict(record=record,conditional_source_probability=source_probability,
                                  conditional_within_cell_probability=1/count,conditional_record_probability=record_probability,
                                  random_output_position_probability=dn/c['sample_count']*record_probability if c['replacement']=='allow' else None,
                                  inclusion_probability=inclusion,
                                  unavailable_probability_reason=None if c['replacement']=='allow' else 'weighted adaptive without replacement: marginal/inclusion not computed; conditional history probabilities exact',
                                  cell=dict(outcome=outcome,domain=domain,source=chosen,balance_key=balance_key),cell_draw_index=j))
    rng.shuffle(draws)
    seen = collections.Counter()
    for i,d in enumerate(draws):
        seen[d['record']['sample_id']] += 1
        d.update(draw_index=i,duplicate_occurrence=seen[d['record']['sample_id']])
    if c['purpose'] == 'fresh_on_policy':
        require(len(seen) == len(draws), 'fresh group reuse forbidden even with replacement; collect new groups separately')
        from schedule import Clock
        from learning import evaluate
        import copy
        learning_config = json.loads(pinned(c['learning_config_ref'],base))
        combined = copy.deepcopy(draws[0]['record']['behavior_bundle'])
        combined['groups'] = []
        combined_clock = Clock(learning_config,draws[0]['record']['clock_state'])
        combined_clock.state['groups'] = {}
        for draw in draws:
            record = draw['record']
            group = record['behavior_bundle']['groups'][0]
            require(group['group_id'] not in combined_clock.state['groups'], 'duplicate fresh behavior group identity')
            require(record['clock_state']['successful_optimizer_steps'] == combined_clock.state['successful_optimizer_steps'], 'mixed successful-step cohorts')
            combined['groups'].append(copy.deepcopy(group))
            combined_clock.state['groups'][group['group_id']] = copy.deepcopy(record['clock_state']['groups'][group['group_id']])
        combined['global_query_count'] = len(draws)
        evaluate(combined,combined_clock)
    counts = collections.Counter(d['record']['outcome'] for d in draws)
    stats = dict(samples=len(draws),unique_records=len(seen),duplicate_draws=len(draws)-len(seen),
                 record_multiplicity=dict(seen),outcomes=dict(counts),
                 recording_reference_multiplicity=dict(collections.Counter(d['record']['recording_ref_id'] for d in draws)),
                 text_source_multiplicity=dict(collections.Counter(d['record']['source_id'] for d in draws)),
                 domains=dict(collections.Counter(d['record']['domain'] for d in draws)),
                 sources=dict(collections.Counter(d['record']['source'] for d in draws)),
                 population=len(outcomes),population_outcomes=dict(collections.Counter(o['status'] for o in outcomes.values())),
                 exclusions=excluded,cells=cells,scope=c['scope'],purpose=c['purpose'])
    return draws,stats


def validate_loader(c,files,synthetic,base):
    loader = c['loader']
    if loader['kind'] == 'missing':
        return dict(status='MISSING_MODEL_TRAINER_LOADER',model_training_ready=False,consumer=loader['consumer'])
    require(loader['kind'] == 'model_trainer_plugin', 'numeric oracle/JSON parsing is not a model trainer loader')
    raw = pinned(loader['module_ref'],base)
    # This plugin is the configured actual consumer interface; never import parent numeric oracle as a loader.
    namespace = {'__name__':'configured_training_consumer','__file__':loader['module_ref']['path']}
    exec(compile(raw,loader['module_ref']['path'],'exec'),namespace)
    require(namespace.get('CONSUMER_KIND') == 'model_trainer' and namespace.get('CONSUMER_ID') == loader['consumer'], 'wrong consumer kind/identity')
    report = namespace[loader['function']](files=files,config=loader['config'])
    require(report.get('loaded_count') == files['count'] and report.get('required_fields_validated') is True
            and report.get('consumer') == loader['consumer'], 'actual consumer failed schema/load validation')
    return dict(status='SYNTHETIC_CONSUMER_INTERFACE_VALIDATED' if synthetic else 'MODEL_CONSUMER_VALIDATED',
                model_training_ready=not synthetic and report.get('actual_model_trainer_loader') is True,report=report)


def publish(c,base,output):
    pool = json.loads(pinned(c['pool_ref'],base))
    draws,stats = sample(pool,c,base)
    implementation = {n:__import__('hashlib').sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ('dataset.py','learning.py','scoring.py','schedule.py','kernel.py')}
    implementation['train_population.py'] = __import__('hashlib').sha256(POPULATION_MODULE.read_bytes()).hexdigest()
    version = digest(dict(config=c,pool_sha256=c['pool_ref']['sha256'],implementation=implementation,order=[d['record']['sample_id'] for d in draws]))
    output = Path(output)/version
    runtime.storage_check(output)
    if (output/'COMPLETE.json').exists():
        complete = json.loads((output/'COMPLETE.json').read_bytes())
        manifest_bytes = (output/'manifest.json').read_bytes()
        require(complete['version'] == version and complete['manifest_sha256'] == __import__('hashlib').sha256(manifest_bytes).hexdigest(), 'existing completion mismatch')
        manifest = json.loads(manifest_bytes)
        require(manifest['version'] == version and manifest['config'] == c and manifest['implementation'] == implementation, 'existing dataset version mismatch')
        for item in manifest['files'].values():
            pinned(item,base)
        return manifest
    def commit(name,value):
        path=output/name
        if path.exists():
            require(json.loads(path.read_bytes()) == value, 'partial release differs; preserve failed attempt')
        else:
            runtime.atomic_json(path,value)
    # Online files never include offline reference labels.
    import copy
    online = [dict(draw_index=d['draw_index'],sample_id=d['record']['sample_id'],online=d['record']['online'],
                   behavior_bundle=copy.deepcopy(d['record'].get('behavior_bundle'))) for d in draws]
    group_supervision = []
    for row in online:
        supervision = {}
        if row['behavior_bundle']:
            for group in row['behavior_bundle']['groups']:
                for episode in group['episodes']:
                    supervision[episode['episode_id']] = episode.pop('offline')
        group_supervision.append(supervision)
    offline = [dict(draw_index=d['draw_index'],sample_id=d['record']['sample_id'],supervision=d['record']['offline_supervision'],group_supervision=group_supervision[i]) for i,d in enumerate(draws)]
    provenance = [{k:v for k,v in d.items() if k != 'record'} | {k:d['record'][k] for k in ('sample_id','recording_ref_id','source_id','source','domain','outcome')} for d in draws]
    commit('online.json',online)
    commit('offline.json',offline)
    commit('provenance.json',provenance)
    files = dict(online=str(output/'online.json'),offline=str(output/'offline.json'),provenance=str(output/'provenance.json'),count=len(draws))
    files['refs']={name:dict(path=files[name],sha256=__import__('hashlib').sha256(Path(files[name]).read_bytes()).hexdigest(),bytes=Path(files[name]).stat().st_size) for name in ('online','offline','provenance')}
    synthetic = pool['synthetic_fixture']
    loader = validate_loader(c,files,synthetic,base)
    ready = loader['model_training_ready'] and not synthetic and c['purpose'] == 'fresh_on_policy'
    manifest = dict(schema='dataset-release-v1',version=version,status='train_ready' if ready else 'analysis_only',
                    training_ready=ready,synthetic_fixture=synthetic,implementation=implementation,config=c,lineage=pool['lineage'],statistics=stats,
                    loader_validation=loader,files={k:dict(path=v,sha256=__import__('hashlib').sha256(Path(v).read_bytes()).hexdigest(),bytes=Path(v).stat().st_size) for k,v in files.items() if k in ('online','offline','provenance')},
                    probability_semantics='exact outcome/domain quotas; seeded source then recording; history-conditional probabilities exact; replacement marginals/inclusion exact; adaptive without-replacement marginals/inclusion explicitly null; not policy importance weights',
                    missing_for_real_training=['actual model trainer loader validation','complete fresh model behavior groups with modality/action/probability fields; screen publication currently analysis only'] if not ready else [])
    commit('manifest.json',manifest)
    runtime.atomic_json(output/'COMPLETE.json',dict(version=version,manifest_sha256=__import__('hashlib').sha256((output/'manifest.json').read_bytes()).hexdigest(),status=manifest['status']))
    return manifest
