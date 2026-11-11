"""Shared dependency-mount contract for both formal bind and formal run."""
from pathlib import Path

from .configured_s1 import require


def readonly_dependency_mounts(mounts, protected_targets):
    result, targets = [], set(protected_targets)
    for mount in mounts:
        require(set(mount) == {'source', 'target'}, 'explicit source/target mount required')
        source, target = Path(mount['source']), Path(mount['target'])
        require(source.is_absolute() and target.is_absolute(), 'absolute dependency mount required')
        require(source.is_dir(), 'readonly dependency source missing')
        require(str(target) not in targets, 'dependency mount shadows execution root or another dependency')
        require(not any(c in str(p) for p in (source, target) for c in (',', '\n', '\r')), 'invalid mount path')
        targets.add(str(target))
        result += ['--mount', f'type=bind,src={source},dst={target},readonly']
    return result
