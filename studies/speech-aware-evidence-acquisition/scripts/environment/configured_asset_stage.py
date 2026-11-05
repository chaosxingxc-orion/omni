#!/usr/bin/env python3
"""Copy hash-pinned external inputs into a readable, separate state directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def execute(config):
    doc = json.loads(Path(config).read_bytes())
    if (set(doc) != {'schema', 'output_root', 'files'} or doc['schema'] != 'saea-asset-stage-v1'
            or not Path(doc['output_root']).is_absolute() or not isinstance(doc['files'], list)
            or not doc['files']):
        raise ValueError('exact asset stage configuration required')
    root = Path(doc['output_root'])
    names = set()
    for pin in doc['files']:
        if (set(pin) != {'source', 'relative', 'sha256'} or not Path(pin['source']).is_absolute()
                or Path(pin['source']).is_symlink() or not Path(pin['source']).is_file()
                or Path(pin['relative']).is_absolute() or '..' in Path(pin['relative']).parts
                or not pin['relative'] or pin['relative'] in names
                or hashlib.sha256(Path(pin['source']).read_bytes()).hexdigest() != pin['sha256']):
            raise ValueError('asset absent, unsafe, duplicated or changed')
        names.add(pin['relative'])
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    outputs = []
    for pin in doc['files']:
        target = root / pin['relative']
        target.parent.mkdir(parents=True, exist_ok=True)
        with Path(pin['source']).open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        if hashlib.sha256(target.read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('asset changed during copy; incomplete directory retained')
        target.chmod(0o600)
        outputs.append({'relative': pin['relative'], 'sha256': pin['sha256']})
    for path in root.rglob('*'):
        if path.is_dir():
            path.chmod(0o700)
    result = {'schema': doc['schema'], 'files': outputs, 'new_model_calls': 0, 'fit_calls': 0}
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    publish_new(root / 'asset-receipt.json', (json.dumps(result, sort_keys=True, indent=2)+'\n').encode())
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    print(json.dumps(execute(p.parse_args().config), sort_keys=True))
