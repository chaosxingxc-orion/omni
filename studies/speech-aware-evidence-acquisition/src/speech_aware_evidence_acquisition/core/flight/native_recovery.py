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
    selected_source = source
    if source.attempt_id != cell.attempt_id:
        replay = load_native_recovery(declaration,data_root=data_root,plan=target,arm=document['source']['arm'])
        selected_source = replay.scientific_source_plan
    if (source.run_id != cell.run_id or target.run_id != cell.run_id
            or selected_source.attempt_id != cell.attempt_id or source.attempt_id == target.attempt_id
            or cell.document.get('resume_from') is not None
            or cell.document.get('checkpoint_compatibility') is not None):
        raise ValueError('native recovery must resume the selected registered cell')
    return CampaignCell.from_document({**cell.document, 'attempt_id': target.attempt_id})


def native_contact_custody_enabled(values):
    """Operational capture for the reviewed odd40 method or raw-vLLM A0 P/S.

    Do not widen this to tool, revision, branch or acquisition graphs merely
    because their numeric ceiling happens to be two. No model inputs change.
    """
    if (values.get('transport') == 'vllm'
            and values.get('arm') == 'prepared-context-text'
            and values.get('result_kind') == 'artifact'
            and values.get('scorers') == []
            and isinstance(values.get('tfgrpo_learning_registration'), dict)
            and isinstance(values.get('tfgrpo_contact_fit'), dict)
            and isinstance(values.get('prepared_training_feedback'), dict)
            and values['prepared_training_feedback'].get('schema') == 'saea-prepared-tfgrpo-dynamic-v1'):
        return True  # Exact registration/source/FIT are independently preflighted.
    native = values.get('native_episode')
    if not isinstance(native, dict):
        return False
    if native.get('answer_selection') == 'odd40-whole-answer-v1':
        return True  # Preserve the already reviewed method's existing path.
    paired = False
    paired_ceiling = 3
    if values.get('policy') in ('phase-native-module-v1-c3','phase-native-module-v1-c4'):
        components=native.get('consumption_module',{}).get('components',[])
        learned=[c for c in components if c.get('component_id')=='tfgrpo-solver-experience'
                 and c.get('state')=='on' and c.get('params',{}).get('target_arm')=='learned']
        if len(learned)!=1:return False
        params=learned[0]['params']
        try:
            from ...rails.tfgrpo_consumption import validate_params,paired_layout_matches
            validate_params(params)
            paired_ceiling=4 if 'compact_params' in params else 3
            paired=paired_layout_matches(values,params) and values.get('policy')==f'phase-native-module-v1-c{paired_ceiling}'
        except (KeyError,TypeError,ValueError):return False
        if not paired:return False
    if (values.get('transport') != 'vllm'
            or values.get('arm') != 'raw-speech-agent-native-tools'
            or values.get('policy') != (f'phase-native-module-v1-c{paired_ceiling}' if paired else 'phase-native-module-v1-c2')
            or values.get('carrier_lock_key') != 'slurp'
            or native.get('review_loop') != 'off'
            or native.get('max_revisions') != 0
            or values.get('phase_tools')
            or any(native.get(k) for k in ('answer_selection', 'acquisition', 'relisten'))):
        return False
    roles = native.get('roles')
    if not isinstance(roles, dict) or set(roles) != {'perception', 'solver', 'reviewer', 'planner'}:
        return False
    for role, expected in (('perception', 1), ('solver', 1), ('reviewer', 0), ('planner', 0)):
        block = roles[role]
        if (not isinstance(block, dict) or block.get('phase_id') != role
                or type(block.get('max_iterations')) is not int
                or block['max_iterations'] != expected
                or block.get('tool_names') or block.get('allowed_tools')
                or block.get('samples', 1) != 1
                or any(k in block for k in ('sampling_variants', 'supply_variants',
                                            'decoding_variants', 'branches'))
                or ('template_variants' in block and not (paired and role=='solver'))):
            return False
    from ...rails.controllers import default_phase_program
    from ...rails.native_episode_config import load_native_episode_config
    try:
        config = load_native_episode_config(values, default_phase_program(values['policy']))
    except (ValueError, TypeError, KeyError):
        return False
    return (config.max_contacts == (paired_ceiling if paired else 2) and config.max_revisions == 0
            and config.acquisition_role is None and config.relisten_json is None
            and all(not role.tool_names and role.samples == 1 for role in config.roles))
