"""Resolve exact reviewed runtime members from the node tree or its parent."""
from pathlib import Path

# Logical name -> path inside the transition checkout. No path from a local
# deployment or caller can expand this list of sibling components.
PARENT_MEMBERS = (
    'collection-dispatch-integration/base.py',
    'collection-dispatch-integration/pool_union.py',
    'collection-dispatch-integration/sampling.py',
    'collection-scale-successor/code/stream_pool.py',
    'collection-scale-successor/code/train_population.py',
    'p-card-field-repair/repair.py',
    'p-card-field-repair/export.py',
    'unified-learning/dataset.py',
)


def source_path(root, name):
    root=Path(root).resolve()
    path=Path(name)
    if path.is_absolute() or '..' in path.parts or str(path)!=name or not path.parts:
        raise ValueError('invalid source member')
    if path.parts[0]=='@transition':
        relative=path.relative_to('@transition').as_posix()
        if relative not in PARENT_MEMBERS:
            raise ValueError('unregistered transition member')
        root=root.parent
        path=Path(relative)
    result=(root/path).resolve(strict=True)
    if not result.is_file() or not result.is_relative_to(root):
        raise ValueError('source escapes checkout')
    return result
