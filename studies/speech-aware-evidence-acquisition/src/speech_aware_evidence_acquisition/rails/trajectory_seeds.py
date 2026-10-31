"""Deterministic per-row Solver trajectory seeds shared with offline plans.

The opt-in recipe lives in the hashed decoding declaration and receipt. Recipe
metadata never reaches the transport; it resolves to one ordinary integer seed.
"""
import hashlib
from collections.abc import Mapping

RECIPE_KEYS = frozenset({'trajectory_seed_base', 'trajectory_index'})
MAX_SEED = 2**31 - 2


def _hash_seed(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16) % MAX_SEED + 1


def iteration_seed(base_seed, iteration):
    return _hash_seed(f'tfrl-v2-iteration|{int(base_seed)}|{int(iteration)}')


def derive_seed(base_seed, row_id, k):
    return _hash_seed(f'tfrl-v2-seed|{int(base_seed)}|{row_id}|{int(k)}')


def recipe_error(value):
    if not isinstance(value, Mapping): return 'trajectory decoding must be an object'
    present = set(value) & RECIPE_KEYS
    if not present: return None
    if present != RECIPE_KEYS: return 'trajectory seed recipe requires both base and index'
    base, index = value['trajectory_seed_base'], value['trajectory_index']
    if type(base) is not int or not 1 <= base <= MAX_SEED:
        return 'trajectory_seed_base must be an integer in the seed domain'
    if type(index) is not int or not 0 <= index < 4:
        return 'trajectory_index must be an integer from 0 through 3'
    temperature = value.get('temperature', 0)
    if type(temperature) not in (int, float) or not 0 < temperature <= 2:
        return 'trajectory sampling requires an explicit positive temperature'
    return None


def group_error(overrides):
    recipes = [v for v in overrides if set(v) & RECIPE_KEYS]
    if not recipes: return None
    if len(recipes) != 4 or len(overrides) != 4:
        return 'seed-only group requires exactly four candidate trajectories'
    if {v['trajectory_index'] for v in recipes} != set(range(4)):
        return 'seed-only group requires each trajectory index exactly once'
    common = [{k:v for k,v in r.items() if k != 'trajectory_index'} for r in recipes]
    if any(r != common[0] for r in common[1:]):
        return 'seed-only trajectories must share all decoding fields and the seed base'
    return None


def resolve(override, row_id):
    """Resolve only a validated recipe; preserve legacy overrides unchanged."""
    reason = recipe_error(override)
    if reason: raise ValueError(reason)
    result = dict(override)
    if not (set(result) & RECIPE_KEYS): return result
    if not isinstance(row_id, str) or not row_id: raise ValueError('trajectory requires a sample identity')
    base, index = result.pop('trajectory_seed_base'), result.pop('trajectory_index')
    result['seed'] = derive_seed(base, row_id, index)
    return result


def verify_recorded_group(row_id, arms, requests):
    """Check seed-only declarations against actual requests, including reasks.

    The caller separately verifies hashes and receipt custody. Legacy groups
    stay unchanged. Each actual request must retain its declared row seed;
    first requests share the lock's evidence and model-facing instruction.
    """
    import json
    overrides = [a.get('decoding_override') for a in arms if a.get('decoding_override') is not None]
    if not any(set(d) & RECIPE_KEYS for d in overrides):return 0
    for d in overrides:
        reason=recipe_error(d)
        if reason:raise ValueError(reason)
    reason=group_error(overrides)
    if reason:raise ValueError(reason)
    indexed={a['contact_id']:[] for a in arms}
    for request in requests:
        if request.get('sample_id') != row_id:continue
        history=request.get('history',[])
        if not history or history[-1].get('role')!='control':continue
        contact=history[-1].get('contact_id')
        if contact in indexed:indexed[contact].append(request)
    if any(not rows for rows in indexed.values()):raise ValueError('seed group lacks actual arm requests')
    lock=indexed[arms[-1]['contact_id']][0]
    fields=('speech_feature_parts','speech_packet_sha256','query_source','supplied_evidence',
            'task_instruction','tool_schemas')
    def context(request):
        history=request['history']
        control=json.loads(history[-1]['content'])
        control.pop('contact_id',None)
        return history[:-1],control
    def provenance(request):
        # rails-supply relpath is a request-owned trace address, not evidence.
        # Validate its exact ownership before normalizing that routing field.
        sources=[]
        for index, source in enumerate(request.get('evidence_provenance', [])):
            if source.get('lock_key') == 'rails-supply':
                if source.get('relpath') != f"rails-message/{request['request_id']}/{index}":
                    raise ValueError('seed group supply provenance has wrong request ownership')
                source={**source,'relpath':f'rails-message/<request>/{index}'}
            sources.append(source)
        return sources
    common=context(lock)
    common_provenance=provenance(lock)
    count=0
    for arm in arms:
        rows=indexed[arm['contact_id']]
        override=arm.get('decoding_override')
        expected=resolve(override,row_id) if override is not None else None
        for request in rows:
            decoding=request.get('decoding_params',{})
            if set(decoding)&RECIPE_KEYS:raise ValueError('seed recipe leaked to transport')
            if expected is not None and any(decoding.get(k)!=v for k,v in expected.items()):
                raise ValueError('actual trajectory decoding differs from declared row seed')
            # Reasks may turn thinking off but must keep the seed and sampling.
            if expected is None and any(decoding.get(k)!=lock['decoding_params'].get(k)
                    for k in ('seed','temperature','top_p','top_k')):
                raise ValueError('twin or lock sampling changed')
            count+=1
        first=rows[0]
        if (any(first.get(k)!=lock.get(k) for k in fields) or context(first)!=common
                or provenance(first)!=common_provenance):
            raise ValueError('seed-only trajectory changed the actual context')
        target={**lock['decoding_params'],**(expected or {})}
        if first['decoding_params']!=target:
            raise ValueError('seed-only first request changed unvaried decoding')
    return count
