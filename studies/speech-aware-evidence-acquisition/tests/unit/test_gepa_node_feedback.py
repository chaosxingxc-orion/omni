import copy
import json

from speech_aware_evidence_acquisition.rails.gepa_mutation_source import node_feedback


def answer(intent, entities):
    return {'scenario': intent[0], 'action': intent[1],
            'entities': [{'type': t, 'filler': f} for t, f in entities]}


def test_separates_wrong_intent_from_delivery_removal():
    raw = answer(('qa', 'factoid'), [('date', 'this year')])
    delivered = answer(('qa', 'factoid'), [])
    gold = answer(('datetime', 'query'), [('date', 'this year')])
    row = {'raw_response': json.dumps(raw), 'delivered_answer': json.dumps(delivered), 'gold_train': gold}
    before = copy.deepcopy(row); result = node_feedback(row)
    assert row == before
    assert not result['raw_vs_gold']['intent_exact']
    assert result['raw_vs_gold']['missing_entities'] == []
    assert result['delivered_vs_gold']['missing_entities'] == [{'type': 'date', 'filler': 'this year', 'count': 1}]
    assert result['delivery_delta']['removed_entities'] == result['delivered_vs_gold']['missing_entities']


def test_valid_normalization_is_difference_not_automatically_a_defect():
    raw = answer(('datetime', 'query'), [('date', '29th March')])
    gold = answer(('datetime', 'query'), [('date', 'twenty ninth march')])
    result = node_feedback({'raw_response': raw, 'delivered_answer': gold, 'gold_train': gold})
    assert result['delivered_vs_gold']['missing_entities'] == []
    assert result['delivery_delta']['removed_entities']
    assert result['causal_attribution'].startswith('unproven')


def test_duplicate_entities_and_unparseable_nodes_not_silently_normalized():
    a = answer(('a', 'b'), [('date', 'today'), ('date', 'today')])
    b = answer(('a', 'b'), [('date', 'today')])
    result = node_feedback({'raw_response': a, 'delivered_answer': b, 'gold_train': b})
    assert result['delivery_delta']['removed_entities'][0]['count'] == 1
    invalid = node_feedback({'raw_response': '```json {} ```', 'delivered_answer': b, 'gold_train': b})
    assert not invalid['parseable']['raw'] and 'delivery_delta' not in invalid
