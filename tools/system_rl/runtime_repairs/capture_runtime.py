"""Resolve and validate the bounded continuation's capture settings."""
import math
from pathlib import Path


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError('relative capture path must be a string')
    path = Path(value)
    if path.is_absolute() or not path.parts or '..' in path.parts or str(path) != value:
        raise ValueError(f'invalid relative capture path: {value}')
    return path


def capture_runtime(spec):
    generation = spec['generation']
    mounts = spec['capture_mounts']
    config = generation['runtime']
    if config['schema'] != 'omni-bounded-capture-runtime-v1':
        raise ValueError('unknown capture runtime configuration')
    targets = {role: Path(mounts['logical_roots'][role])
               for role in ('receipts', 'cold', 'overlays', 'tools', 'index')}
    reserved = tuple(map(Path, ('/sealed', '/omni-hot', '/prior-checkpoint',
                               '/usr', '/opt', '/etc', '/dev', '/proc', '/sys')))
    for role, target in targets.items():
        if not target.is_absolute() or len(target.parts) < 2 or '..' in target.parts:
            raise ValueError(f'invalid {role} mount target')
        peers = [p for key, p in targets.items() if key != role] + list(reserved)
        if any(target.is_relative_to(p) or p.is_relative_to(target) for p in peers):
            raise ValueError(f'overlapping {role} mount target')
    sources = config['sources']
    required = {'publication', 'bounded_tp4', 'fresh_adapter', 'finalize_capture',
                'capture_source', 'cohort_source'}
    if set(sources) != required:
        raise ValueError('incomplete capture source roles')
    for source in sources.values():
        if relative_path(source).suffix != '.py' or source not in mounts['overlays']:
            raise ValueError(f'capture source is not pinned: {source}')
    if len(set(sources.values())) != len(required):
        raise ValueError('capture source roles must be distinct')
    cache_bytes = config['prepared_cache_bytes']
    if type(cache_bytes) is not int or cache_bytes < 0:
        raise ValueError('invalid prepared packet cache capacity')
    parity = generation['parity']
    for name in ('token_atol', 'token_rtol', 'sequence_atol'):
        value = parity[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid parity threshold: {name}')
    return dict(cohort_dir=str(relative_path(config['cohort_dir'])),
                prepared_cache_bytes=cache_bytes, targets=targets, index_root=str(targets['index']),
                **{name: str(targets['overlays']/path) for name, path in sources.items()})
