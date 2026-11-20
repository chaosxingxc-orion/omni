"""Compare staged runtime files with the immutable Git revision in the release lock."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def verify(repo, lock, roots):
    commit = lock['source_commit']
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('an immutable source commit is required')
    roles = lock['roles']
    if not roles or set(roots) != set(roles):
        raise ValueError('provide every configured deployment role')
    rows = []
    for role, paths in roles.items():
        if not paths:
            raise ValueError('empty component roster')
        for deployed, source in paths.items():
            for name in (deployed, source):
                path = Path(name)
                if not path.parts or path.is_absolute() or '..' in path.parts:
                    raise ValueError('expected relative component paths')
            revision = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{source}'])
            expected = hashlib.sha256(revision).hexdigest()
            path = Path(roots[role]) / deployed
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            rows.append(dict(role=role, component=deployed, expected=expected, observed=actual,
                             matched=actual == expected))
    return dict(source_commit=commit, source_matches=all(row['matched'] for row in rows),
                components=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'lock', 'roots'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.repo, json.loads(args.lock.read_bytes()), json.loads(args.roots.read_bytes()))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['source_matches'] else 1)
