"""Assemble pinned real H1 material for the existing external slot compiler; offline only."""
import argparse
import hashlib
import json
from pathlib import Path

from external_slot_candidate import checked, read
from speech_aware_evidence_acquisition.rails.semantic_knowledge_edits import slot_value_edit_contract


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    if set(cfg) != {'schema', 'authored', 'records', 'pool', 'forbidden_sample_ids',
                    'slot_ids', 'fields', 'max_edits', 'max_field_chars', 'output'} or cfg['schema'] != 'saea-ace-slot-recipe-v1':
        raise ValueError('exact ACE slot recipe config required')
    authored, records, pool, forbidden = (read(checked(cfg[k])) for k in
                                          ('authored', 'records', 'pool', 'forbidden_sample_ids'))
    if authored.get('schema') != 'saea-authored-semantic-knowledge-source-v1':
        raise ValueError('authored source required')
    rows = authored['rows']
    if not isinstance(records, list) or not records:
        raise ValueError('real H1 construction records required')
    groups = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('source_id'), str):
            raise ValueError('record source_id required')
        group = record['source_id']
        ids = record.get('source_sample_ids')
        if group in groups or not isinstance(ids, list) or not ids or any(
                not isinstance(s, str) or not s.startswith(group + ':') for s in ids):
            raise ValueError('unique source groups with complete prefixed ancestry required')
        groups[group] = ids
    identity = lambda s: ':'.join(s.split(':')[:3]) if s.startswith('slurp:') else s
    if not isinstance(forbidden, list) or any(not isinstance(s, str) for s in forbidden):
        raise ValueError('forbidden sample IDs must be a list of strings')
    if {identity(s) for ids in groups.values() for s in ids} & {identity(s) for s in forbidden}:
        raise ValueError('construction evidence intersects forbidden samples')
    contract = slot_value_edit_contract(rows, slot_ids=cfg['slot_ids'], fields=cfg['fields'],
        evidence_groups=groups, max_edits=cfg['max_edits'], max_field_chars=cfg['max_field_chars'])
    dynamic = {'pool': pool, 'records': records, 'base_rows': rows, 'edit_contract': contract}
    out = Path(cfg['output']).resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    write(out / 'base.json', rows)
    write(out / 'contract.json', contract)
    write(out / 'dynamic.json', dynamic)
    write(out / 'messages.json', [{'role': 'user', 'content': json.dumps(dynamic, ensure_ascii=False, allow_nan=False)}])
    write(out / 'pins.json', {name: pin(out / (name + '.json')) for name in ('base', 'contract', 'dynamic', 'messages')})
    print(json.dumps({'state': 'prepared-only', 'output': str(out)}))


if __name__ == '__main__':
    main()
