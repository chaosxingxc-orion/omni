"""Per-contact numbered candidates projected from actual pinned P evidence."""
import json


def bind(messages, values, *, input_k, output_k):
    rows, seen = [], set()
    for part in messages[1]['content']:
        if not isinstance(part, dict) or part.get('knowledge_role') != 'value':
            continue
        value_id = part.get('evidence_id', '')
        if not value_id.startswith('intent-boundaries::') or value_id not in values:
            raise ValueError('ranking input contains an unauthorized knowledge source')
        value = values[value_id]
        if part.get('text') != value.content:
            raise ValueError('ranking card differs from pinned knowledge')
        prefix = 'intent-boundary:'
        local = value_id.split('::', 1)[1]
        if not local.startswith(prefix) or not local[len(prefix):]:
            raise ValueError('ranking card lacks an intent identity')
        intent = local[len(prefix):]
        if value.task_target is not None and value.task_target != intent:
            raise ValueError('ranking label differs from pinned target')
        if intent in seen:
            continue
        seen.add(intent)
        rows.append({'index': len(rows)+1, 'intent_id': intent, 'knowledge_id': value_id,
                     'description': value.content})
    if len(rows) != input_k or output_k > len(rows):
        raise ValueError('ranking actual candidate count differs from configured window')
    return rows


def response_format(count, output_k):
    return {'type': 'json_schema', 'json_schema': {'name': 'ranked_candidate_indices', 'strict': True,
        'schema': {'type': 'array', 'minItems': output_k, 'maxItems': output_k,
                   'items': {'type': 'integer', 'enum': list(range(1, count+1))}}}}


def parse(text, rows, output_k):
    try:
        indices = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('ranking output must be a JSON array') from error
    if (not isinstance(indices, list) or len(indices) != output_k
            or any(type(i) is not int or not 1 <= i <= len(rows) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('ranking indices violate the bound window')
    return [rows[i-1] for i in indices]


def semantic_projection(rows):
    # Internal adapter for the existing state/handoff contract; never model output.
    return json.dumps({'candidates': [{'intent_id': r['intent_id'], 'distinction': '',
        'evidence_refs': [r['knowledge_id']], 'unresolved_fields': []} for r in rows]},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'))
