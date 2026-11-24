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
    mount_roles = {'receipts', 'cold', 'overlays', 'tools', 'index'}
    if 'policy_assets' in mounts['logical_roots']:
        mount_roles.add('policy_assets')
    if spec.get('formal_sampled'):
        mount_roles.update(mounts.get('formal_input_roles', []))
    if set(mounts['logical_roots']) != mount_roles or any(
            name in ('', '.', '..') or Path(name).name != name for name in mount_roles):
        raise ValueError('unexpected capture mount roles')
    targets = {role: Path(mounts['logical_roots'][role]) for role in sorted(mount_roles)}
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
    if config.get('reference_batch') is not None:
        required.add('reference_batch_source')
    if {'consumer_source','consumer_wrapper'} & set(sources):
        required.update(('consumer_source','consumer_wrapper'))
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
    reuse_scores = config.get('reuse_group_current_scores', False)
    if type(reuse_scores) is not bool:
        raise ValueError('reuse_group_current_scores must be boolean')
    encoding=config.get('row_encoding','original')
    if encoding not in ('original','compact_json_v1'):raise ValueError('unknown cohort row encoding')
    actor_fields=config.get('actor_fields')
    if actor_fields is not None and (actor_fields!='reuse_accepted_v1' or not spec.get('formal_sampled') or
                                     not config.get('reference_batch')):
        raise ValueError('accepted field reuse requires a formal batched reference')
    layout = config.get('group_preparation', 'replicated')
    if layout not in ('replicated','shard_by_draw_v1') or (layout == 'shard_by_draw_v1' and
            (not spec.get('formal_sampled') or not config.get('reference_batch') or actor_fields != 'reuse_accepted_v1')):
        raise ValueError('sharded capture requires formal batch reference and accepted actor fields')
    batch=config.get('reference_batch')
    if batch is not None and (not isinstance(batch,dict) or set(batch)!={'rows_per_rank','padded_tokens_per_rank'}
            or any(type(v) is not int or v<1 for v in batch.values())):
        raise ValueError('invalid frozen reference batch limits')
    parity = generation['parity']
    for name in ('token_atol', 'token_rtol', 'sequence_atol'):
        value = parity[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid parity threshold: {name}')
    return dict(cohort_dir=str(relative_path(config['cohort_dir'])),
                prepared_cache_bytes=cache_bytes, reuse_group_current_scores=reuse_scores,actor_fields=actor_fields,row_encoding=encoding,group_preparation=layout,
                **({'reference_batch':batch} if batch is not None else {}),
                targets=targets, index_root=str(targets['index']),
                **{name: str(targets['overlays']/path) for name, path in sources.items()})
