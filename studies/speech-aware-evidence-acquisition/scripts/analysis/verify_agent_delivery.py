"""Read-only verification of task-bound child artifacts; never launches or retries."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def load(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj
    return json.loads(path.read_text(), object_pairs_hook=pairs)


def verify(directory):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('delivery directory required')
    task = root / 'TASK.md'
    if not task.is_file() or task.is_symlink():
        raise ValueError('regular TASK.md required')
    expected = digest(task)
    receipt_path, result_path = root/'receipt.json', root/'result.json'
    if receipt_path.is_symlink():
        raise ValueError('receipt symlink forbidden')
    if not receipt_path.exists():
        return {'state': 'no-receipt', 'task_sha256': expected,
                'meaning': 'task execution unverified; not proof of permission failure'}
    if receipt_path.is_symlink():
        raise ValueError('receipt symlink forbidden')
    receipt = load(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError('receipt JSON object required')
    if receipt.get('task_sha256') != expected:
        return {'state': 'stale-receipt', 'task_sha256': expected}
    if result_path.is_symlink():
        raise ValueError('result symlink forbidden')
    if not result_path.exists():
        return {'state': 'received-no-result', 'task_sha256': expected,
                'meaning': 'inspect actual process/tool state; no automatic retry'}
    if result_path.is_symlink():
        raise ValueError('result symlink forbidden')
    result = load(result_path)
    if not isinstance(result, dict):
        raise ValueError('result JSON object required')
    if result.get('task_sha256') != expected:
        return {'state': 'stale-result', 'task_sha256': expected}
    if result.get('status') not in ('delivered', 'blocked'):
        raise ValueError('explicit result status required')
    artifacts = result.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('nonempty artifacts required')
    names = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {'path', 'sha256'}:
            raise ValueError('exact artifact pin required')
        if not isinstance(artifact['path'], str) or not artifact['path']:
            raise ValueError('nonempty artifact path required')
        path = Path(artifact['path'])
        if '..' in path.parts:
            raise ValueError('artifact parent traversal forbidden')
        path = path if path.is_absolute() else root/path
        # Disallow symlink path components, even if their destination is inside root.
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved == result_path:
            raise ValueError('artifact must belong to this delivery directory')
        relative = path.absolute().relative_to(root)
        current = root
        for part in relative.parts:
            current = current/part
            if current.is_symlink():
                raise ValueError('artifact symlink forbidden')
        if not resolved.is_file() or resolved in names:
            raise ValueError('unique regular artifact required')
        names.add(resolved)
        if digest(resolved) != artifact['sha256']:
            raise ValueError('artifact bytes changed')
    if root/'REPORT.md' not in names:
        raise ValueError('pinned complete report required')
    return {'state': ('blocked' if result['status'] == 'blocked' else 'artifacts-verified'),
            'reported_status': result['status'],
            'task_sha256': expected, 'artifact_count': len(names),
            'business_acceptance': 'pending root review and independent reproduction',
            'automatic_retry': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.directory)
        result['automatic_retry'] = False
        print(json.dumps(result, ensure_ascii=False))
        if result['state'] == 'artifacts-verified':
            return 0
        return 4 if result['state'] == 'blocked' else 3
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'state': 'invalid-delivery', 'error': str(error),
                          'automatic_retry': False}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
