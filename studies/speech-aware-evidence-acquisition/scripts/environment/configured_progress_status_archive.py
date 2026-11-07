#!/usr/bin/env python3
"""Bounded local archive of existing observer status JSON; never read checkpoints."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def load_config(path):
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    if set(doc) != {'schema', 'interval_seconds', 'max_snapshots', 'output_dir', 'sources'}:
        raise ValueError('exact archive configuration required')
    if doc['schema'] != 'saea-progress-status-archive-v1' or doc['interval_seconds'] != 600:
        raise ValueError('unsupported schema or interval')
    if type(doc['max_snapshots']) is not int or not 1 <= doc['max_snapshots'] <= 145:
        raise ValueError('bounded snapshots required')
    if not Path(doc['output_dir']).is_absolute() or not isinstance(doc['sources'], dict) or not doc['sources']:
        raise ValueError('absolute output and named sources required')
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not Path(v).is_absolute()
           for k, v in doc['sources'].items()):
        raise ValueError('source names and absolute paths required')
    return doc, hashlib.sha256(raw).hexdigest()


def capture(sources, *, now=None):
    result = {'schema': 'saea-progress-status-snapshot-v1',
              'archived_at_utc': (now or dt.datetime.now(dt.timezone.utc)).isoformat(),
              'sources': {}}
    for name, path in sources.items():
        try:
            raw = Path(path).read_bytes()
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError('status is not an object')
            result['sources'][name] = {'path': path, 'state': 'available',
                'sha256': hashlib.sha256(raw).hexdigest(),
                'observed_at_utc': document.get('observed_at_utc'), 'document': document}
        except (OSError, UnicodeError, ValueError) as error:
            result['sources'][name] = {'path': path, 'state': 'unavailable',
                                       'error_type': type(error).__name__}
    return result


def run(path):
    config, pin = load_config(path)
    root = Path(config['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'archive.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in range(config['max_snapshots']):
            target = root / f'{index:04d}.json'
            snapshot = capture(config['sources'])
            snapshot['configuration_sha256'] = pin
            with target.open('x', encoding='utf-8') as output:
                json.dump(snapshot, output, indent=2, sort_keys=True, allow_nan=False)
                output.write('\n')
                output.flush()
                os.fsync(output.fileno())
            if index + 1 < config['max_snapshots']:
                time.sleep(config['interval_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
