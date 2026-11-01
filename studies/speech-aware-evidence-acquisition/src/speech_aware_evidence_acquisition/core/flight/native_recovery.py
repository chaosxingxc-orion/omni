"""Load a pinned operational recovery receipt without changing model protocol."""
from __future__ import annotations

import json
from pathlib import Path

from ...contracts import ExecutionPlan
from ..contact_journal import NativeContactReplay, digest


def load_native_recovery(declaration, *, data_root, plan, arm):
    if not isinstance(declaration, dict) or set(declaration) != {'path', 'sha256'}:
        raise ValueError('native recovery requires a pinned receipt path and sha256')
    path = Path(declaration['path'])
    if not path.is_absolute() or not path.resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError('native recovery receipt must be confined to data root')
    blob = path.read_bytes()
    if digest(blob) != declaration['sha256']:
        raise ValueError('native recovery receipt bytes changed')
    document = json.loads(blob)
    if (set(document) != {'schema', 'source', 'target_plan'}
            or document['schema'] != 'saea-native-recovery-v1'
            or ExecutionPlan(**document['target_plan']) != plan):
        raise ValueError('native recovery receipt names another target plan')
    source = document['source']
    replay = NativeContactReplay(data_root, ExecutionPlan(**source['plan']), plan, arm)
    if replay.source_binding() != source:
        raise ValueError('native recovery predecessor changed')
    return replay


def select_recovery_cell(cell, declaration, *, data_root):
    """Select only a fresh attempt of the registered cell, preserving fragments.

    The receipt changes operational attempt identity, never the scientific
    config hash. Preflight independently verifies the entire target plan.
    """
    from .campaign import CampaignCell
    path = Path(declaration['path'])
    if not path.is_absolute() or not path.resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError('native recovery receipt must be confined to data root')
    blob = path.read_bytes()
    if digest(blob) != declaration['sha256']:
        raise ValueError('native recovery receipt bytes changed')
    document = json.loads(blob)
    if document.get('schema') != 'saea-native-recovery-v1':
        raise ValueError('unsupported native recovery receipt')
    source = ExecutionPlan(**document['source']['plan']).validate()
    target = ExecutionPlan(**document['target_plan']).validate()
    if (source.run_id != cell.run_id or target.run_id != cell.run_id
            or source.attempt_id != cell.attempt_id or source.attempt_id == target.attempt_id
            or cell.document.get('resume_from') is not None
            or cell.document.get('checkpoint_compatibility') is not None):
        raise ValueError('native recovery must resume the selected registered cell')
    return CampaignCell.from_document({**cell.document, 'attempt_id': target.attempt_id})
