"""Verify a prepared TP4 packet against an immutable parent and explicit deltas."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
        return h.hexdigest()


def snapshot(root):
    # Deployment addresses, boot IDs and receipt locations belong to each host.
    # Hash executable sources, bundled model assets, effective rows and engine options.
    lanes = sorted(root.glob('lane-*/config.json'))
    if not lanes:
        raise ValueError('missing prepared lane')
    values = {}
    for lane in lanes:
        config = read(lane)
        engine = {key: config[key] for key in ('image', 'served_model_name', 'serve_args',
                                                 'client_chat_template_kwargs')}
        server = read(root / 'bundle' / ('SERVER-' + lane.parent.name[5:] + '.json'))
        engine.update(release_kind=server['release_kind'], policy_sha256=server['policy_ref']['sha256'])
        for f in lane.parent.glob('*.py'):
            engine[f.name] = digest(f)
        if values and engine != values:
            raise ValueError('heterogeneous lane runtime')
        values = engine
    bundle = root / 'bundle'
    files = {str(f.relative_to(bundle)): digest(f) for f in bundle.rglob('*')
             if f.is_file() and '__pycache__' not in f.parts and
             (f.suffix == '.py' or 'checkpoint' in f.relative_to(bundle).parts or
              f.parent != bundle and f.suffix == '.json' or
              f.name in ('PRECISION-SPEC.json', 'PROFILE.json'))}
    return {'engine': values, 'rows': read(bundle / 'ROWS.json'), 'files': files}


def changes(parent, child):
    return {section: {key: {'old': before.get(key), 'new': after.get(key)}
                      for key in sorted(set(before) | set(after))
                      if key not in before or key not in after or before[key] != after[key]}
            for section, before in parent.items() for after in [child[section]]}


def verify(root, lock_path, declaration):
    if declaration['parent_sha256'] != digest(lock_path):
        raise ValueError('incorrect decode parent lock hash')
    lock = read(lock_path)
    delta = changes(lock['runtime'], snapshot(root))
    if declaration['changes'] != delta:
        raise ValueError('prepared runtime differs from declared parent deltas')
    return {'parent': lock['version'], 'parent_sha256': declaration['parent_sha256'], 'changes': delta}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--declaration', type=Path)
    args = parser.parse_args()
    if args.declaration:
        result = verify(args.packet, args.lock, read(args.declaration))
    else:
        result = {'parent_sha256': digest(args.lock),
                  'changes': changes(read(args.lock)['runtime'], snapshot(args.packet))}
    print(json.dumps(result, indent=2, sort_keys=True))
