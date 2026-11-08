"""Score P outputs against a separately supplied, ordered evaluation roster.

This is deliberately a scorer, not a model executor or a system score. Actor
inputs must be constructed independently of the gold-bearing roster.
"""
from collections import Counter
from collections.abc import Mapping, Sequence

from ..rails.intent_slot_pipeline import document, normalize_evidence_references
from ..rails.adapters.slurp_intent_slot import validate_response
from ..rails.candidate_handoff import normalized_response


def score(roster, outputs, *, maximum, legal_ids):
    if not isinstance(roster, Sequence) or isinstance(roster, (str, bytes)):
        raise ValueError('ordered roster required')
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise ValueError('output records required')
    if type(maximum) is not int or maximum < 1 or not legal_ids:
        raise ValueError('candidate limit and carrier inventory required')
    ids = [r['sample_id'] for r in roster]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('roster sample IDs must be unique')
    by_id = {}
    for row in outputs:
        if isinstance(row, Mapping) and row.get('result_kind') == 'node':
            node = row.get('node_outcome')
            if not isinstance(node, Mapping) or node.get('sample_id') != row.get('sample_id') or node.get('response') != row.get('text'):
                raise ValueError('node output does not bind its persisted row')
            row = {**node, 'text': node['response']}
        if not isinstance(row, Mapping) or row.get('sample_id') not in ids:
            raise ValueError('output outside pinned roster')
        by_id.setdefault(row['sample_id'], []).append(row)
    records = []
    for item in roster:
        sid, gold = item['sample_id'], item['gold_intent']
        if not isinstance(gold, str) or gold not in legal_ids:
            raise ValueError('gold intent outside carrier inventory')
        matches = by_id.get(sid, [])
        status, candidate_ids, raw_duplicate = 'valid', [], False
        if not matches:
            status = 'missing'
        elif len(matches) != 1:
            status = 'duplicate_output'
        elif matches[0].get('status') == 'invalid_output':
            # The native protocol may reject an otherwise parseable response
            # (for example, on request-bound evidence). Never promote it by
            # re-parsing the text without the full original execution state.
            status = 'invalid'
        elif matches[0].get('status') == 'finite_completion':
            status = 'incomplete'
        elif matches[0].get('status') not in (None, 'ok'):
            status = 'invalid'
        elif matches[0].get('finish_reason') != 'stop':
            status = 'incomplete'
        else:
            try:
                output = matches[0]
                config = output['pipeline_config']
                if (config['max_candidates'] != maximum
                        or config.get('duplicate_candidate_policy') != 'group_v2'
                        or not config['schema'].endswith('-v2')):
                    raise ValueError('unexpected native pipeline configuration')
                if config.get('ranking_window'):
                    from ..rails.ranked_candidate_indices import parse
                    ranking = output['ranking_result']
                    selected = parse(output['text'], ranking['input'], min(maximum, len(ranking['input'])) if config['ranking_window'].get('dynamic') else maximum)
                    candidate_ids = [row['intent_id'] for row in selected]
                    if candidate_ids != ranking['candidate_ids'] or any(cid not in legal_ids for cid in candidate_ids):
                        raise ValueError('ranking mapping differs from registered output')
                    records.append({'sample_id': sid, 'gold_intent': gold, 'status': status,
                                    'candidate_ids': candidate_ids, 'raw_duplicate_candidates': False,
                                    'gold_recalled': gold in candidate_ids})
                    continue
                raw = document(output['text'])
                raw_ids = [row['intent_id'] for row in raw['candidates']]
                raw_duplicate = len(set(raw_ids)) != len(raw_ids)
                context = {'evidence_inventory': output['evidence_inventory']}
                if context['evidence_inventory']['sample_id'] != sid:
                    raise ValueError('evidence inventory belongs to another sample')
                effective = output['text']
                if config.get('reference_normalizer'):
                    effective, _ = normalize_evidence_references(
                        effective, context['evidence_inventory'], [],
                        policy=config.get('evidence_policy', 'strict'),
                        normalizer=config['reference_normalizer'])
                rows = validate_response(kind='perception', text=effective,
                                         response_format=output['response_format'],
                                         context=context, config=config)
                candidate_ids = [v['candidate_id'] for v in rows]
                if any(cid not in legal_ids for cid in candidate_ids):
                    raise ValueError('candidate outside carrier inventory')
                normalized_response('perception', effective, context, config)
            except KeyError:
                status = 'missing_native_context'
            except (ValueError, TypeError, AttributeError):
                status = 'invalid'
        records.append({'sample_id': sid, 'gold_intent': gold, 'status': status,
                        'candidate_ids': candidate_ids,
                        'raw_duplicate_candidates': raw_duplicate,
                        'gold_recalled': status == 'valid' and gold in candidate_ids})
    n = len(records)
    counts = Counter(r['gold_intent'] for r in records)
    hits = Counter(r['gold_intent'] for r in records if r['gold_recalled'])
    ranks = [r['candidate_ids'].index(r['gold_intent']) + 1 for r in records if r['gold_recalled']]
    ranked = [by_id[r['sample_id']][0].get('ranking_result') for r in records if len(by_id.get(r['sample_id'], [])) == 1]
    supplied = [r for r in records if len(by_id.get(r['sample_id'], [])) == 1
                and isinstance(by_id[r['sample_id']][0].get('ranking_result'), Mapping)
                and r['gold_intent'] in [v['intent_id'] for v in by_id[r['sample_id']][0]['ranking_result']['input']]]
    return {'schema': 'saea-perception-node-score-v1',
            'node_complete': all(len(by_id.get(sid, [])) == 1 for sid in ids),
            'system_complete': False, 'rows': records, 'denominator': n,
            'valid_output_count': sum(r['status'] == 'valid' for r in records),
            'raw_duplicate_candidate_rows': sum(r['raw_duplicate_candidates'] for r in records),
            'gold_recall_count': sum(r['gold_recalled'] for r in records),
            'top1_count': sum(r == 1 for r in ranks),
            'reciprocal_rank_sum': sum(1 / r for r in ranks),
            **({'ranking_input_gold_count': len(supplied),
                'ranking_input_retained_count': sum(r['gold_recalled'] for r in supplied)} if any(ranked) else {}),
            'candidate_count': sum(len(r['candidate_ids']) for r in records),
            'per_intent': {k: {'denominator': v, 'recall_count': hits[k]}
                           for k, v in sorted(counts.items())}}


def paired(control, candidate):
    a, b = control['rows'], candidate['rows']
    if [(r['sample_id'], r['gold_intent']) for r in a] != [(r['sample_id'], r['gold_intent']) for r in b]:
        raise ValueError('paired scores must share the same ordered roster and gold')
    return {'denominator': len(a),
            'repairs': sum(not x['gold_recalled'] and y['gold_recalled'] for x, y in zip(a, b)),
            'regressions': sum(x['gold_recalled'] and not y['gold_recalled'] for x, y in zip(a, b))}
