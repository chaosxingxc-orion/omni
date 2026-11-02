"""Read a FINALIZED synthesis as input to a new, explicitly distinct experiment.

This does not reopen a successful attempt or call the failed-final repair API on
it. Its successful final contact and all recursively imported costs are retained.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path
import sys

from speech_aware_evidence_acquisition.rails.tfgrpo_learning import LearningError, sha, validate_experience, parse_operations
from speech_aware_evidence_acquisition.rails.tfgrpo_consumption import checked_pin, verify_learning_seal
from speech_aware_evidence_acquisition.rails.tfgrpo_final_import import load_import
from .grounded_synthesis_plan import bound_units


def load_source(spec, *, data_root):
    """Verify source seal, original stages, proposal correspondence and all costs."""
    if not isinstance(spec, dict) or set(spec) != {'schema', 'candidate', 'source_manifest', 'source_batch', 'operations'}:
        raise LearningError('exact finalized synthesis source required')
    if spec['schema'] != 'saea-grounded-synthesis-source-v1':
        raise LearningError('unsupported synthesis source schema')
    root = Path(data_root).resolve()
    paths = {k: checked_pin(root, spec[k]) for k in ('candidate', 'source_manifest', 'source_batch', 'operations')}
    candidate = json.loads(paths['candidate'].read_bytes())
    batch = json.loads(paths['source_batch'].read_bytes())
    operations = json.loads(paths['operations'].read_bytes())
    validate_experience(candidate, batch['model_id'])
    if (paths['candidate'].name != 'candidate.json'
            or paths['source_batch'] != paths['candidate'].parent / 'batch.json'
            or batch.get('schema') != 'saea-tfgrpo-batch-v1'
            or candidate.get('batch_sha256') != sha(batch)
            or not batch.get('final_import')
            or candidate.get('status') != 'candidate'
            or candidate.get('deployment_accepted') is not False):
        raise LearningError('finalized synthesis batch/candidate binding differs')
    # The shared verifier checks the final attempt/trace, native response and FIT.
    bundle = verify_learning_seal(spec, data=root, candidate=candidate)
    imported, historical = load_import(batch['final_import'], data_root=root,
        **{k: batch[k] for k in ('groups', 'common', 'model_id', 'parent', 'policy_sha256', 'rollout_kind')})
    if historical != candidate.get('historical_stage_custody'):
        raise LearningError('candidate historical costs or stage custody differ')
    units = bound_units(batch, operations, expected_groups=len(batch['groups']),
                        expected_operations=len(imported))
    projected = [{k: row[k] for k in ('operation', 'id', 'content')}
                 for unit in units for row in unit['operations']]
    from collections import Counter
    if Counter(sha(row) for row in projected) != Counter(sha(row) for row in imported):
        raise LearningError('planned operations differ from verified native group updates')
    # Bind each proposal to its exact group response, including operation index.
    # The import reader above has verified the full group response journal.
    cursor = historical
    while cursor.get('historical_final_custody') is not None:
        cursor = cursor['historical_final_custody']
    origin = root / 'runs' / cursor['source_run_id'] / cursor['source_attempt_id'] / 'tfgrpo/learner/contacts'
    import hashlib
    for unit in units:
        response = origin / (unit['group_contact'] + '.response.json')
        response_bytes = response.read_bytes()
        digest = hashlib.sha256(response_bytes).hexdigest()
        response_doc = json.loads(response_bytes)
        parsed = parse_operations(response_doc['response']['text'], batch['parent']['items'],
                                  max_operations=batch['bounds']['max_operations'])
        actual_group = [{k: row[k] for k in ('operation', 'id', 'content')} for row in unit['operations']]
        if actual_group != parsed:
            raise LearningError('proposal content differs from its native group response')
        if any(row['response_file_sha256'] != digest for row in unit['operations']):
            raise LearningError('proposal group response differs from verified native source')
    accounting = candidate.get('accounting', {})
    current = accounting.get('known_usage')
    count = candidate.get('planned_learning_calls')
    if (type(count) is not int or count < 1 or accounting.get('unknown_contacts') != 0
            or accounting.get('attempted_contacts') != count or accounting.get('observed_contacts') != count
            or current != candidate.get('learning_usage') or not isinstance(current, dict)
            or current.get('calls') != count):
        raise LearningError('finalized source contact accounting incomplete')
    prior = historical['original_total_usage']
    totals = {}
    for k in ('calls', 'prompt_tokens', 'completion_tokens'):
        if any(type(v.get(k)) is not int or v[k] < 0 for v in (prior, current)):
            raise LearningError('nonnegative exact historical usage required')
        totals[k] = prior[k] + current[k]
    # Final-import native_source_usage already includes recursive final imports.
    # Summary imports have a separate ledger and must be added exactly once.
    ledger = copy.deepcopy(historical['native_source_usage'])
    cursor = historical
    while cursor is not None:
        summary = cursor.get('historical_summary_custody')
        if summary is not None:
            ledger.append({'run_id': summary['source_run_id'],
                           'attempt_id': summary['source_attempt_id'],
                           'usage': copy.deepcopy(summary['original_total_usage'])})
        cursor = cursor.get('historical_final_custody')
    attempt = bundle.manifest['attempt']
    ledger.append({'run_id': attempt['run_id'], 'attempt_id': attempt['attempt_id'],
                   'usage': copy.deepcopy(current)})
    seen, ledger_total = set(), dict.fromkeys(totals, 0)
    for entry in ledger:
        identity = (entry.get('run_id'), entry.get('attempt_id'))
        if not all(isinstance(v, str) and v for v in identity) or identity in seen:
            raise LearningError('duplicate or missing source spend identity')
        seen.add(identity)
        usage = entry.get('usage')
        if not isinstance(usage, dict) or set(usage) != set(totals):
            raise LearningError('complete native source usage required')
        for k, value in usage.items():
            if type(value) is not int or value < 0:
                raise LearningError('nonnegative exact native source usage required')
            ledger_total[k] += value
    if ledger_total != totals:
        raise LearningError('native source ledger drops or duplicates historical cost')
    result = {'schema': 'saea-grounded-synthesis-source-custody-v1', 'source': spec,
              'source_native_custody_verified': True, 'source_attempt_state': bundle.attempt_state,
              'source_batch_sha256': spec['source_batch']['sha256'],
              'source_operations_sha256': spec['operations']['sha256'],
              'source_candidate_sha256': candidate['sha256'], 'model_id': batch['model_id'],
              'groups': len(units), 'operations': len(imported),
              'prior_stage_usage': prior, 'successful_synthesis_usage': current,
              'original_total_usage': totals,
              'native_source_usage': ledger, 'unknown_contacts': 0,
              'historical_custody_sha256': sha(historical), 'new_model_calls': 0,
              'successful_source_reopened': False, 'generation_authorized': False}
    result['custody_sha256'] = sha(result)
    return result
