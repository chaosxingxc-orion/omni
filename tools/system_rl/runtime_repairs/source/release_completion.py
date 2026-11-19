"""Shared release publication boundary for producers and client consumers.

Legacy sealed releases have neither marker. New producers publish intent before
copying and completion only after durable_tree. A DELTA file alone is not ready.
Existing source-content validation is still required after this lifecycle check.
"""
import hashlib
import json
from pathlib import Path


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def require_closed_release(root):
    root = Path(root).resolve()
    delta = root / 'DELTA.json'
    assembly = root / 'ASSEMBLY.json'
    intent = root / 'BUILD-INTENT.json'
    marker = root / 'BUILD-COMPLETE.json'
    if not delta.is_file() or not assembly.is_file():
        raise ValueError('release manifests incomplete: ' + str(root))
    required = json.loads(delta.read_text()).get('build_completion_required', False)
    if not required and not intent.exists() and not marker.exists():
        return dict(protocol='legacy', persistence_receipt=False)
    if not intent.is_file() or not marker.is_file():
        raise ValueError('release persistence incomplete: ' + str(root))
    value = json.loads(marker.read_text())
    expected = dict(schema='durable-release-complete-v1',
                    build_intent_ref=pin(intent), delta_ref=pin(delta),
                    assembly_ref=pin(assembly))
    if value != expected:
        raise ValueError('release commit identity drift: ' + str(root))
    return dict(protocol=value['schema'], persistence_receipt=True, completion_ref=pin(marker))
