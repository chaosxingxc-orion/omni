"""Read-only inventory of an existing launch config; never grants readiness.

Directory probes cover only directory access, not recursive file availability.
File pins are streamed and checked separately. No environment values are emitted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

import configured_job_launch as launcher


def probe(path, expected=None):
    result = {'path': str(path), 'readable': False}
    try:
        info = path.stat()
        if stat.S_ISDIR(info.st_mode):
            result['kind'] = 'directory'
            if expected is not None:
                raise ValueError('pinned input must be a file')
            with os.scandir(path) as entries:
                next(entries, None)
            result['coverage'] = 'directory access only; descendants not verified'
        elif stat.S_ISREG(info.st_mode):
            result.update(kind='file', size_bytes=info.st_size)
            with path.open('rb') as handle:
                if expected is None:
                    handle.read(1)
                    result['coverage'] = 'first byte only; content identity not verified'
                else:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                    result['sha256'] = digest.hexdigest()
                    result['expected_sha256'] = expected
                    result['identity_matches'] = result['sha256'] == expected
                    result['coverage'] = 'complete file hash at observation'
        else:
            result['kind'] = 'unsupported'
            raise ValueError('ordinary file or directory required')
        result['readable'] = True
    except (OSError, ValueError) as error:
        result.update(error_type=type(error).__name__, errno=getattr(error, 'errno', None))
    return result


def inventory(config):
    doc = launcher.load_config(config)
    mounts = [{**probe(Path(m['source'])), 'target': m['target'], 'readonly': m['readonly']}
              for m in doc['mounts']]
    pins = [probe(Path(p['path']), p['sha256']) for p in doc['required_inputs']]
    failures = [r for r in mounts + pins
                if not r['readable'] or r.get('identity_matches') is False]
    return {'schema': 'saea-configured-input-inventory-v1',
            'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'launch_config_sha256': launcher.digest(config),
            'container_name': doc['container_name'], 'mounts': mounts, 'required_inputs': pins,
            'blocked_checks': len(failures), 'readiness_granted': False, 'new_model_calls': 0,
            'limitations': ['Not a recursive asset manifest or storage authorization check.',
                           'Readable directories do not establish readable or byte-identical datasets.',
                           'No launch, copying, remount, retries, or runtime-state changes.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = inventory(args.config)
    # Never replace historical evidence, including concurrent writers.
    launcher.persist(args.output, result)
    print(json.dumps({'blocked_checks': result['blocked_checks'], 'report': str(args.output),
                      'readiness_granted': False, 'new_model_calls': 0}))
    return 1 if result['blocked_checks'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
