"""Restore immutable receipt addresses from the preboot serving archive.

Verify every archived byte before writing any address. Never recreate a resident
session marker or change archived receipts, manifests, or current services.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def verified_receipts(archive):
    manifest = json.loads((archive / 'manifest.json').read_bytes())
    records = manifest['records']
    if not records: raise ValueError('empty historical archive')
    receipts = []
    for name, record in records.items():
        if Path(name).name != name or name in ('.', '..'): raise ValueError('unsafe session name')
        files = record['files']
        if 'session-receipt.json' not in files: raise ValueError('missing receipt')
        for rel, digest in files.items():
            path = (archive / name / rel).resolve()
            if not path.is_relative_to((archive / name).resolve()): raise ValueError('archive escape')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest: raise ValueError('archive bytes changed')
            if rel == 'session-receipt.json' or rel.startswith('session-receipt.json.archive/'):
                if rel != 'session-receipt.json' and path.name != digest + '.json':
                    raise ValueError('content address differs')
                receipts.append((name, raw))
    return receipts


def restore(data, archive, out):
    import fcntl
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.session.receipt import store_session_receipt_bytes
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    if (not archive.resolve().is_relative_to(data.resolve() / 'work/codex')
            or not out.resolve().is_relative_to(data.resolve() / 'work/codex') or out.exists()):
        raise ValueError('owned archive and fresh output required')
    with (data / '.vllm-resident-registration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = verified_receipts(archive)
        for name, raw in receipts:
            store_session_receipt_bytes(data / 'serving' / name / 'session-receipt.json', raw)
        result = {'restored_receipt_addresses': len(receipts), 'new_model_calls': 0,
            'resident_receipts_rewritten': False,
            'archive_manifest_sha256': hashlib.sha256((archive / 'manifest.json').read_bytes()).hexdigest()}
        publish_new(out, (json.dumps(result, indent=2) + '\n').encode())
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'archive', 'out'): p.add_argument('--' + key, type=Path, required=True)
    print(json.dumps(restore(**vars(p.parse_args()))))
