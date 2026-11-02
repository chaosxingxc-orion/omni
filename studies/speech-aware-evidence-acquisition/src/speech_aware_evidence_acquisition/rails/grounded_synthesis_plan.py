"""CPU-only deterministic TRAIN evidence batching; never admits model contacts.

Every group and its complete proposal set travels together. Token counts use an
injected counter over complete chat messages, including the generation prefix.
The CLI uses the local model tokenizer/template; these are text counts, not FIT.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    LearningError, build_final_messages, canonical, parse_operations, sha,
)

POLICY = 'evidence-grounded-synthesis-v1'
SCHEMA = 'saea-grounded-synthesis-batch-plan-v1'
HEX = re.compile(r'[0-9a-f]{64}')


def _require(ok, message):
    if not ok:
        raise LearningError(message)


def _pin(blob, expected, label):
    _require(HEX.fullmatch(expected or '') is not None, label + ' SHA256 required')
    _require(hashlib.sha256(blob).hexdigest() == expected, label + ' bytes differ')
    return json.loads(blob)


def bound_units(batch, operations, *, expected_groups, expected_operations):
    """Bind operation hashes/indices to exact group identities, without inference."""
    groups = batch['groups']
    _require(batch.get('schema') == 'saea-tfgrpo-batch-v1', 'source batch schema differs')
    _require(isinstance(groups, list) and len(groups) == expected_groups, 'source group count differs')
    _require(isinstance(operations, list) and len(operations) == expected_operations,
             'source operation count differs')
    units = []
    by_contact = {}
    for position, group in enumerate(groups):
        qid = group.get('query_id')
        _require(isinstance(qid, str) and qid.startswith('slurp:train:'), 'TRAIN groups only')
        contact = sha({'stage': 'group_update', 'identity': qid})
        _require(contact not in by_contact, 'duplicate TRAIN group')
        unit = {'group_index': position, 'group_sha256': sha(group), 'group_contact': contact,
                'operation_indices': [], 'operations': [], 'group': group}
        units.append(unit)
        by_contact[contact] = unit
    seen = set()
    for index, row in enumerate(operations):
        contact, ordinal = row['group_contact'], row['operation_index']
        _require(contact in by_contact, 'proposal group is absent from TRAIN source')
        _require(type(ordinal) is int and ordinal >= 0, 'invalid operation index')
        _require((contact, ordinal) not in seen, 'duplicate proposal identity')
        seen.add((contact, ordinal))
        operation = {key: row[key] for key in ('operation', 'id', 'content')}
        _require(sha(operation) == row['operation_sha256'], 'proposal content hash differs')
        _require(HEX.fullmatch(row.get('response_file_sha256', '')) is not None,
                 'proposal response hash required')
        unit = by_contact[contact]
        unit['operation_indices'].append(index)
        unit['operations'].append(row)
    response_owners = set()
    for unit in units:
        rows = unit['operations']
        _require(rows and [row['operation_index'] for row in rows] == list(range(len(rows))),
                 'group proposal indices must be complete and ordered')
        _require(len({row['response_file_sha256'] for row in rows}) == 1,
                 'group proposals disagree on source response')
        response_hash = rows[0]['response_file_sha256']
        _require(response_hash not in response_owners, 'source response reused across TRAIN groups')
        response_owners.add(response_hash)
        parse_operations(json.dumps([{k: row[k] for k in ('operation', 'id', 'content')}
                                     for row in rows]), batch['parent']['items'],
                         max_operations=expected_operations)
    return units


def _messages(batch, units):
    operations = [{key: row[key] for key in ('operation', 'id', 'content')}
                  for unit in units for row in unit['operations']]
    common = {**batch['common'], 'final_merge_policy': POLICY,
              'final_merge_limits': {'max_items': 128, 'max_characters': 16000}}
    proposal_sources = [{'proposal_index': index, 'query_id': unit['group']['query_id'],
                         'group_contact': unit['group_contact'],
                         'source_operation_index': row['operation_index'],
                         'operation_sha256': row['operation_sha256']}
                        for index, (unit, row) in enumerate(
                            (unit, row) for unit in units for row in unit['operations'])]
    material = json.dumps({'existing_experiences': batch['parent']['items'],
                           'proposed_operations': operations, 'proposal_sources': proposal_sources},
                          ensure_ascii=False, sort_keys=True)
    return build_final_messages(common, material, 128, 16000,
                                groups=[unit['group'] for unit in units])


def plan_batches(batch_bytes, operation_bytes, *, source_batch_sha256, source_operations_sha256,
                 count_tokens, tokenizer_identity, expected_groups=51, expected_operations=102,
                 resident_context=49152, completion_cap=16384, safety_margin=1024):
    """Stable contiguous greedy packing; no truncation, filtering or model call."""
    for value in (expected_groups, expected_operations, resident_context, completion_cap):
        _require(type(value) is int and value > 0, 'positive integer limits required')
    _require(type(safety_margin) is int and safety_margin >= 0, 'invalid safety margin')
    input_limit = resident_context - completion_cap - safety_margin
    _require(input_limit > 0, 'no room for input tokens')
    _require(isinstance(tokenizer_identity, dict) and tokenizer_identity,
             'tokenizer identity required')
    batch = _pin(batch_bytes, source_batch_sha256, 'source batch')
    operations = _pin(operation_bytes, source_operations_sha256, 'source operations')
    units = bound_units(batch, operations, expected_groups=expected_groups,
                        expected_operations=expected_operations)

    def measured(selected):
        messages = _messages(batch, selected)
        tokens = count_tokens(messages)
        _require(type(tokens) is int and tokens > 0, 'counter must return positive integer tokens')
        return messages, tokens

    shards, pending = [], []

    def append(selected, messages, tokens):
        group_indices = [unit['group_index'] for unit in selected]
        operation_indices = [index for unit in selected for index in unit['operation_indices']]
        shards.append({'shard_index': len(shards), 'group_indices': group_indices,
                       'group_sha256': [unit['group_sha256'] for unit in selected],
                       'group_contacts': [unit['group_contact'] for unit in selected],
                       'operation_indices': operation_indices,
                       'operation_sha256': [operations[i]['operation_sha256'] for i in operation_indices],
                       'response_file_sha256': [unit['operations'][0]['response_file_sha256'] for unit in selected],
                       'messages': messages, 'messages_sha256': sha(messages), 'input_tokens': tokens,
                       'completion_cap': completion_cap, 'safety_margin': safety_margin,
                       'total_reserved_tokens': tokens + completion_cap + safety_margin})

    previous_messages, previous_tokens = None, None
    for unit in units:
        single_messages, single_tokens = measured([unit])
        _require(single_tokens <= input_limit,
                 'one complete TRAIN group exceeds context; no truncation permitted')
        messages, tokens = measured(pending + [unit])
        if tokens <= input_limit:
            pending.append(unit)
            previous_messages, previous_tokens = messages, tokens
            continue
        if pending:
            append(pending, previous_messages, previous_tokens)
        pending = [unit]
        previous_messages, previous_tokens = single_messages, single_tokens
    if pending:
        append(pending, previous_messages, previous_tokens)
    group_indices = [i for shard in shards for i in shard['group_indices']]
    operation_indices = [i for shard in shards for i in shard['operation_indices']]
    _require(group_indices == list(range(expected_groups)), 'group coverage differs')
    _require(sorted(operation_indices) == list(range(expected_operations)), 'operation coverage differs')
    result = {'schema': SCHEMA, 'status': 'CPU_PLAN_ONLY_NOT_ADMITTED', 'policy': POLICY,
              'packing': 'source-order-contiguous-greedy-whole-groups-v1',
              'source_batch_sha256': source_batch_sha256,
              'source_operations_sha256': source_operations_sha256,
              'tokenizer_identity': tokenizer_identity, 'groups': expected_groups,
              'operations': expected_operations, 'resident_context': resident_context,
              'completion_cap': completion_cap, 'safety_margin': safety_margin,
              'input_token_limit': input_limit, 'shards': shards,
              'coverage': {'complete': True, 'duplicate_groups': 0, 'duplicate_operations': 0},
              'new_model_calls': 0, 'h2_read': False, 'test_read': False,
              'generation_authorized': False, 'native_fit_completed': False,
              'source_native_custody_verified': False,
              'cross_batch_consolidation': {'status': 'NOT_IMPLEMENTED',
                                            'generation_authorized': False},
              'limitations': ['Token counts use the complete local text chat template; native FIT remains required.',
                             'Evidence uses the existing explicitly lossy Perception/feedback projection.',
                             'Source byte pins and group mappings do not replace historical native custody verification.',
                             'Only independent synthesis shards are planned; final consolidation requires a separate bounded, evidence-bound protocol.']}
    result['plan_sha256'] = sha(result)
    return result


def verify_plan(plan, batch_bytes, operation_bytes, *, count_tokens, tokenizer_identity):
    """Rebuild from source pins and compare every byte-level semantic field."""
    expected = plan_batches(batch_bytes, operation_bytes,
                            source_batch_sha256=plan['source_batch_sha256'],
                            source_operations_sha256=plan['source_operations_sha256'],
                            count_tokens=count_tokens, tokenizer_identity=tokenizer_identity,
                            expected_groups=plan['groups'], expected_operations=plan['operations'],
                            resident_context=plan['resident_context'], completion_cap=plan['completion_cap'],
                            safety_margin=plan['safety_margin'])
    _require(plan == expected, 'batch plan differs from deterministic source reconstruction')
    return True


def public_summary(plan):
    """No raw prompts, query identities, references, operations or private paths."""
    keys = ('schema', 'status', 'policy', 'packing', 'source_batch_sha256', 'source_operations_sha256',
            'plan_sha256', 'groups', 'operations', 'resident_context', 'completion_cap', 'safety_margin',
            'input_token_limit', 'coverage', 'new_model_calls', 'h2_read', 'test_read',
            'generation_authorized', 'native_fit_completed', 'source_native_custody_verified',
            'cross_batch_consolidation', 'limitations')
    result = {key: plan[key] for key in keys}
    result['tokenizer_identity_sha256'] = sha(plan['tokenizer_identity'])
    result['shards'] = [{'shard_index': row['shard_index'], 'groups': len(row['group_indices']),
                         'operations': len(row['operation_indices']), 'input_tokens': row['input_tokens'],
                         'completion_cap': row['completion_cap'], 'safety_margin': row['safety_margin'],
                         'total_reserved_tokens': row['total_reserved_tokens'],
                         'messages_sha256': row['messages_sha256']} for row in plan['shards']]
    return result


def load_token_counter(tokenizer_path):
    """Use the same local tokenizer identity for planning and runtime validation."""
    from transformers import AutoTokenizer
    tokenizer_path = Path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True)
    template_path = tokenizer_path / 'chat_template.json'
    tokenizer.chat_template = json.loads(template_path.read_bytes())['chat_template']
    names = ('tokenizer.json', 'vocab.json', 'merges.txt', 'tokenizer.model',
             'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.json', 'config.json')
    identity = {name: hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest()
                for name in names if (tokenizer_path / name).is_file()}
    _require('tokenizer.json' in identity or {'vocab.json', 'merges.txt'} <= identity.keys()
             or 'tokenizer.model' in identity, 'tokenizer vocabulary identity missing')
    identity['runtime_versions'] = {name: importlib.metadata.version(name)
                                    for name in ('transformers', 'tokenizers', 'jinja2')}
    identity['tokenizer_class'] = type(tokenizer).__name__
    identity['rendering'] = {'add_generation_prompt': True, 'add_special_tokens': False,
                             'truncation': False}
    def count(messages):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(rendered, add_special_tokens=False, truncation=False))
    return count, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'operations', 'tokenizer', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-batch-sha256', 'source-operations-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    count, identity = load_token_counter(args.tokenizer)
    batch_bytes, operation_bytes = args.batch.read_bytes(), args.operations.read_bytes()
    plan = plan_batches(batch_bytes, operation_bytes, source_batch_sha256=args.source_batch_sha256,
                        source_operations_sha256=args.source_operations_sha256,
                        count_tokens=count, tokenizer_identity=identity)
    verify_plan(plan, batch_bytes, operation_bytes, count_tokens=count, tokenizer_identity=identity)
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    def save(name, blob):
        with (args.out / name).open('xb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
    save('source-batch.json', batch_bytes)
    save('source-operations.json', operation_bytes)
    save('private-plan.json', canonical(plan) + b'\n')
    summary = public_summary(plan)
    save('summary.json', (json.dumps(summary, indent=2) + '\n').encode())
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
