"""Register an observed deployment through a pinned JSON command."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
SCHEMA = 'saea-register-configured-resident-v1'


def load(path):
    doc = json.loads(Path(path).read_bytes())
    if (set(doc) != {'schema', 'repo', 'data', 'model_fragment', 'session_id', 'observations'}
            or doc['schema'] != SCHEMA):
        raise ValueError('exact resident registration config required')
    for key in ('repo', 'data'):
        value = doc[key]
        if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('absolute repository and data paths required')
    for key in ('model_fragment', 'session_id'):
        value = doc[key]
        if not isinstance(value, str) or not value or Path(value).name != value or value in ('.', '..'):
            raise ValueError('confined identity required')
    pin = doc['observations']
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'} or not Path(pin['path']).is_absolute():
        raise ValueError('pinned observation required')
    blob = Path(pin['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != pin['sha256']:
        raise ValueError('observation changed')
    return doc, json.loads(blob)


def execute(path):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.config import read_fragment, resolve_runtime_receipt_path
    from speech_aware_evidence_acquisition.core.session.vllm_resident import register_deployment
    assert_execution_environment()
    doc, observations = load(path)
    repo, data = Path(doc['repo']), Path(doc['data'])
    fragment = read_fragment(repo, 'model', doc['model_fragment'])
    paths = register_deployment(data_root=data, fragment=fragment,
        model_fragment=doc['model_fragment'], receipt_path=resolve_runtime_receipt_path(repo, fragment),
        instances=observations, session_id=doc['session_id'])
    return {'session_id': doc['session_id'], 'files': len(paths), 'new_model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    print(json.dumps(execute(parser.parse_args().config), sort_keys=True))
