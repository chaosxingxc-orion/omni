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
