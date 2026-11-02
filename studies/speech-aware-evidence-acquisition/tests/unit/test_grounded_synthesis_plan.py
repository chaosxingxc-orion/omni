import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location('grounded_batches', ROOT / 'scripts/analysis/grounded_synthesis_plan.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def source():
    groups = [{'query_id': f'slurp:train:{i}', 'query': f'private query {i}',
               'answer': f'private reference {i}',
               'rollouts': [{'critique': 'private feedback', 'reward': 0.5}]} for i in range(4)]
    batch = {'schema': 'saea-tfgrpo-batch-v1', 'groups': groups, 'parent': {'items': {}},
             'common': {'agent_objective': 'Infer intent.', 'learning_objective': 'Learn grounded rules.'}}
    operations = []
    for group in groups:
        for index in range(2):
            op = {'operation': 'ADD', 'id': None, 'content': f'proposal {index}'}
            operations.append({**op, 'group_contact': m.sha({'stage': 'group_update', 'identity': group['query_id']}),
                               'operation_index': index, 'operation_sha256': m.sha(op),
                               'response_file_sha256': m.sha({'source_response': group['query_id']})})
    return batch, operations


def count(messages):
    # Synthetic unit fixture controls group size; production uses real tokenizer.
    evidence = messages[-1]['content'].split('<Bound TRAIN evidence>\n')[1].split('\n</Bound TRAIN evidence>')[0]
    return 100 + 100 * len(json.loads(evidence))


def plan(batch=None, operations=None, **extra):
    original_batch, original_ops = source()
    batch = original_batch if batch is None else batch
    operations = original_ops if operations is None else operations
    bb, ob = m.canonical(batch), m.canonical(operations)
    args = dict(source_batch_sha256=m.hashlib.sha256(bb).hexdigest(),
                source_operations_sha256=m.hashlib.sha256(ob).hexdigest(), count_tokens=count,
                tokenizer_identity={'fixture_only': 'a'*64}, expected_groups=4, expected_operations=8,
                resident_context=500, completion_cap=100, safety_margin=100)
    args.update(extra)
    return m.plan_batches(bb, ob, **args)


def test_complete_deterministic_group_and_operation_coverage():
    result = plan()
    assert result == plan()
    assert [s['group_indices'] for s in result['shards']] == [[0, 1], [2, 3]]
    assert [s['operation_indices'] for s in result['shards']] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert all(s['input_tokens'] + 100 + 100 <= 500 for s in result['shards'])
    assert result['generation_authorized'] is result['native_fit_completed'] is False
    assert result['new_model_calls'] == 0
    assert result['cross_batch_consolidation']['status'] == 'NOT_IMPLEMENTED'
    batch, ops = source()
    assert m.verify_plan(result, m.canonical(batch), m.canonical(ops), count_tokens=count,
                         tokenizer_identity={'fixture_only': 'a'*64})


@pytest.mark.parametrize('field', ['query_id', 'content', 'group_contact', 'operation_index', 'missing'])
def test_source_mapping_and_scope_rejections(field):
    batch, ops = source()
    if field == 'query_id':
        batch['groups'][0]['query_id'] = 'slurp:test:1'
    elif field == 'content':
        ops[0]['content'] = 'changed'
    elif field == 'group_contact':
        ops[0]['group_contact'] = 'f'*64
    elif field == 'operation_index':
        ops[1]['operation_index'] = 0
    else:
        ops.pop()
    with pytest.raises(m.LearningError):
        plan(batch, ops)


def test_single_oversized_group_fails_without_truncation():
    with pytest.raises(m.LearningError, match='one complete TRAIN group exceeds'):
        plan(resident_context=399)


def test_individual_group_checked_even_if_combined_counter_is_smaller():
    def nonmonotonic(messages):
        evidence = messages[-1]['content'].split('<Bound TRAIN evidence>\n')[1].split('\n</Bound TRAIN evidence>')[0]
        rows = json.loads(evidence)
        return 400 if len(rows) == 1 and rows[0]['query_id'].endswith(':1') else 100
    with pytest.raises(m.LearningError, match='one complete TRAIN group exceeds'):
        plan(count_tokens=nonmonotonic)


def test_source_pin_and_invalid_counter_rejected():
    with pytest.raises(m.LearningError, match='source batch bytes differ'):
        plan(source_batch_sha256='f'*64)
    with pytest.raises(m.LearningError, match='positive integer tokens'):
        plan(count_tokens=lambda _: True)


@pytest.mark.parametrize('change', ['token_count', 'duplicate_group', 'omit_operation', 'prompt', 'tokenizer'])
def test_reconstruction_rejects_modified_plan(change):
    result = copy.deepcopy(plan())
    if change == 'token_count':
        result['shards'][0]['input_tokens'] -= 1
    elif change == 'duplicate_group':
        result['shards'][1]['group_indices'][0] = 0
    elif change == 'omit_operation':
        result['shards'][0]['operation_indices'].pop()
    elif change == 'prompt':
        result['shards'][0]['messages'][-1]['content'] += 'changed'
    else:
        result['tokenizer_identity']['fixture_only'] = 'b'*64
    unsigned = {k: v for k, v in result.items() if k != 'plan_sha256'}
    result['plan_sha256'] = m.sha(unsigned)
    batch, ops = source()
    with pytest.raises(m.LearningError, match='deterministic source reconstruction'):
        m.verify_plan(result, m.canonical(batch), m.canonical(ops), count_tokens=count,
                      tokenizer_identity={'fixture_only': 'a'*64})


def test_public_summary_excludes_raw_evidence_and_identities():
    summary = m.public_summary(plan())
    encoded = json.dumps(summary)
    for secret in ('private query', 'private reference', 'private feedback', 'slurp:train:',
                   'proposal 0', 'group_contacts', 'group_indices', 'operation_indices', '"messages"'):
        assert secret not in encoded
    assert sum(row['groups'] for row in summary['shards']) == 4
    assert sum(row['operations'] for row in summary['shards']) == 8


@pytest.mark.parametrize('change', ['duplicate_group', 'duplicate_response', 'mixed_response', 'index_gap'])
def test_duplicate_and_incomplete_sources_rejected(change):
    batch, ops = source()
    if change == 'duplicate_group':
        batch['groups'][1] = copy.deepcopy(batch['groups'][0])
    elif change == 'duplicate_response':
        for row in ops[2:4]:
            row['response_file_sha256'] = ops[0]['response_file_sha256']
    elif change == 'mixed_response':
        ops[1]['response_file_sha256'] = 'f'*64
    else:
        ops[1]['operation_index'] = 2
    with pytest.raises(m.LearningError):
        plan(batch, ops)


def test_prompt_contains_explicit_operation_to_group_mapping():
    result = plan()
    for shard in result['shards']:
        content = shard['messages'][-1]['content']
        material = content.split('<Existing library and candidate evidence>\n')[1].split('\n</Existing library and candidate evidence>')[0]
        projected = json.loads(material)
        assert len(projected['proposal_sources']) == len(projected['proposed_operations'])
        for index, (operation, binding) in enumerate(zip(projected['proposed_operations'], projected['proposal_sources'])):
            assert binding['proposal_index'] == index
            assert binding['operation_sha256'] == m.sha(operation)
            assert binding['group_contact'] == m.sha({'stage': 'group_update', 'identity': binding['query_id']})
