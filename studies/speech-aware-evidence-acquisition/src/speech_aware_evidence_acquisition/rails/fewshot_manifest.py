"""Bind a pinned, per-query frozen candidate list and its original examples."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=8)
def _load(path, sha, kind):
    data = Path(path).read_bytes()
    if _sha(data) != sha:
        raise ValueError(f'{kind} sha256 mismatch: {path}')
    if kind == 'manifest':
        raw = json.loads(data)
        if raw.get('gold_read_for_candidates') is not False:
            raise ValueError('manifest must declare gold-free candidate generation')
        rows = raw['rows']
        result = {row['sample_id']: row for row in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate manifest sample')
        return result
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    result = {row['member_index']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate bank member')
    return result


def prepare(messages, values, snapshot_sha256, sample_id, config):
    """Return separately sourced card messages and label-indexed fewshot evidence."""
    required = {'manifest_path', 'manifest_sha256', 'bank_path', 'bank_sha256', 'fewshots_per_label'}
    if set(config) != required or type(config['fewshots_per_label']) is not int or not 0 <= config['fewshots_per_label'] <= 5:
        raise ValueError('invalid fewshot manifest configuration')
    manifests = _load(config['manifest_path'], config['manifest_sha256'], 'manifest')
    bank = _load(config['bank_path'], config['bank_sha256'], 'bank')
    row = manifests.get(sample_id)
    if row is None:
        raise ValueError(f'fewshot manifest missing sample: {sample_id}')
    excluded = set(row['excluded_source_groups'])
    if not excluded or len(excluded) != len(row['excluded_source_groups']):
        raise ValueError(f'invalid excluded sources: {sample_id}')
    candidates = row['candidates']
    if row['candidate_labels'] != [c['label'] for c in candidates] or len(set(row['candidate_labels'])) != len(candidates):
        raise ValueError(f'duplicate or inconsistent candidates: {sample_id}')
    import copy
    output = copy.deepcopy(messages)
    output[1]['content'] = [p for p in output[1]['content'] if not isinstance(p, dict) or 'evidence_id' not in p]
    evidence = []
    for c in candidates:
        label, kid = c['label'], c['knowledge_id']
        value = values.get(kid)
        if kid != 'intent-boundaries::intent-boundary:' + label or value is None or (value.task_target is not None and value.task_target != label):
            raise ValueError(f'card identity mismatch: {sample_id} {label}')
        if not c['routes'] or any(route not in ('question', 'description') for route in c['routes']):
            raise ValueError(f'invalid candidate route: {sample_id} {label}')
        output[1]['content'].append({'type':'text', 'text':value.content, 'evidence_id':kid,
            'source_uri':f'knowledge-snapshot://{snapshot_sha256}/{kid}', 'knowledge_role':'value'})
        examples, seen = [], set()
        for index in c['fewshot_member_indices_top5'][:config['fewshots_per_label']]:
            item = bank.get(index)
            if item is None or item['intent'] != label or not isinstance(item['text'], str) or not item['text']:
                raise ValueError(f'fewshot member/label mismatch: {sample_id} {label} {index}')
            source = item['source_group']
            if not isinstance(source, str) or not source or source in excluded or source in seen:
                raise ValueError(f'fewshot source excluded/duplicate: {sample_id} {label} {index}')
            if _sha(item['text'].encode()) != item['sentence_sha256']:
                raise ValueError(f'fewshot text hash mismatch: {sample_id} {label} {index}')
            seen.add(source)
            examples.append({'member_index': index, 'source_group': source, 'text': item['text'], 'text_sha256': item['sentence_sha256']})
        evidence.append({'label':label, 'knowledge_id':kid, 'routes':c['routes'],
                         'question_source_rank':c['question_source_rank'], 'description_rank':c['description_rank'], 'examples':examples})
    return output, evidence


def attach(rows, evidence):
    if len(rows) != len(evidence):
        raise ValueError('candidate and fewshot counts differ')
    for row, entry in zip(rows, evidence):
        if (row['intent_id'], row['knowledge_id']) != (entry['label'], entry['knowledge_id']):
            raise ValueError('candidate and fewshot identities differ')
        row['fewshot_evidence'] = entry
    return rows


def render(rows, *, include_empty=True):
    lines = []
    for row in rows:
        lines.append(f"{row['index']}. {row['intent_id']}")
        for example in row['fewshot_evidence']['examples']:
            lines.append('   Example: ' + json.dumps(example['text'], ensure_ascii=False))
        if include_empty and not row['fewshot_evidence']['examples']:
            lines.append('   Examples: none available')
    return '\n'.join(lines)
