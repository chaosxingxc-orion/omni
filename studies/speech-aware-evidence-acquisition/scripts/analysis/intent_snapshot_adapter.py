"""Publish a source-bound, section-only intent snapshot; never contact a model."""
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from p_teacher_proposals import compile_one
from speech_aware_evidence_acquisition.rails.knowledge_stores import load_knowledge_store_binding

SCHEMA = 'saea-intent-section-publication-v1'
POLICY = 'saea-existing-intent-card-complete-section-reorder-v1'


def read_pin(pin):
    if set(pin) != {'path', 'sha256'}:
        raise ValueError('exact input pin required')
    data = Path(pin['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin['sha256']:
        raise ValueError('input pin differs')
    return json.loads(data)


def publish(cfg):
    if set(cfg) != {'schema', 'data_root', 'baseline', 'candidate', 'collected', 'context', 'parent_context', 'source', 'output', 'snapshot_id'} or cfg['schema'] != SCHEMA:
        raise ValueError('exact publication configuration required')
    data = Path(cfg['data_root']).resolve()
    base, candidate, collected, context, parent_context, source = [read_pin(cfg[k]) for k in ('baseline', 'candidate', 'collected', 'context', 'parent_context', 'source')]
    index = candidate['index']
    if type(index) is not int or index < 0 or index >= len(collected['proposals']['candidates']) or candidate['status'] != 'compiled':
        raise ValueError('compiled collected candidate required')
    if candidate['collected_sha256'] != cfg['collected']['sha256'] or candidate['raw_candidate'] != collected['proposals']['candidates'][index] or candidate['raw_response_sha256'] != collected['response']['response_sha256']:
        raise ValueError('candidate custody differs')
    compiled = compile_one(candidate['raw_candidate'], method='reflexion', axis='construction', context=context, parent_context=parent_context, source=source)
    if compiled != candidate['artifact'] or not compiled['patches']:
        raise ValueError('candidate reconstruction differs')
    binding = load_knowledge_store_binding(base, data_root=data)
    store = binding.store('intent-boundaries')
    if store.store_role != 'intent-boundaries' or store.snapshot.source_partition_role != 'train':
        raise ValueError('TRAIN intent-boundaries parent required')
    cards = context['locked_intent_card_bodies']
    values = store.snapshot.values
    if set(cards) != {'intent-boundaries::' + v.value_id for v in values}:
        raise ValueError('complete locked card inventory differs')
    for v in values:
        key = 'intent-boundaries::' + v.value_id
        c = cards[key]
        if c['id'] != key or c['content'] != v.content or c['content_sha256'] != hashlib.sha256(v.content.encode()).hexdigest() or c['source_uri'] != f'knowledge-snapshot://{binding.binding_sha256}/{key}':
            raise ValueError('locked parent card or lineage differs')
    patches = {p['target_id']: p for p in compiled['patches']}
    if len(patches) != len(compiled['patches']) or not set(patches) <= set(cards):
        raise ValueError('invalid targets')
    new_values = tuple(replace(v, content=patches['intent-boundaries::'+v.value_id]['new']) if 'intent-boundaries::'+v.value_id in patches else v for v in values)
    if not isinstance(cfg['snapshot_id'], str) or not cfg['snapshot_id'].strip():
        raise ValueError('snapshot id required')
    child = replace(store.snapshot, snapshot_id=cfg['snapshot_id'], builder_policy_id=POLICY, values=new_values)
    frozen = child.frozen_json()
    root = (data / cfg['output']).resolve()
    if not root.is_relative_to(data) or root == data:
        raise ValueError('output must be confined to data root')
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        path = root / name
        blob = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        with path.open('xb') as f:
            f.write(blob)
        return {'relpath':path.relative_to(data).as_posix(), 'sha256':hashlib.sha256(blob).hexdigest()}
    snapshot_file = write('snapshot.json', frozen)
    snapshot_pin = {'relpath':snapshot_file['relpath'], 'sha256':child.content_hash()}
    release_pin = write('release.json', {'schema':'saea-knowledge-store-release-v1', 'store_id':store.store_id, 'store_role':store.store_role, 'snapshot_sha256':child.content_hash(), 'source_split_identity_hash':child.source_split_identity_hash, 'builder_policy_id':child.builder_policy_id, 'source':dict(store.source_pin)})
    patch = [{'op':'test', 'path':'/knowledge_stores/intent-boundaries/snapshot', 'value':dict(store.snapshot_pin)}, {'op':'test', 'path':'/knowledge_stores/intent-boundaries/release', 'value':dict(store.release_pin)}, {'op':'replace', 'path':'/knowledge_stores/intent-boundaries/snapshot', 'value':snapshot_pin}, {'op':'replace', 'path':'/knowledge_stores/intent-boundaries/release', 'value':release_pin}]
    updated = copy.deepcopy(base)
    updated['knowledge_stores']['intent-boundaries']['snapshot'] = snapshot_pin
    updated['knowledge_stores']['intent-boundaries']['release'] = release_pin
    updated.pop('prepared_phase_supply', None)
    resolved = load_knowledge_store_binding(updated, data_root=data)
    if resolved.store('intent-boundaries').snapshot != child:
        raise ValueError('native child resolution differs')
    audit = {'schema':'saea-intent-section-release-audit-v1', 'parent_snapshot':dict(store.snapshot_pin), 'parent_release':dict(store.release_pin), 'parent_composite_sha256':binding.binding_sha256, 'snapshot':snapshot_pin, 'snapshot_file_sha256':snapshot_file['sha256'], 'release':release_pin, 'source':dict(store.source_pin), 'source_split_identity_hash':child.source_split_identity_hash, 'candidate_sha256':cfg['candidate']['sha256'], 'collected_sha256':cfg['collected']['sha256'], 'context_sha256':cfg['context']['sha256'], 'source_context_sha256':cfg['source']['sha256'], 'changed_cards':sorted(patches), 'card_count':len(values), 'unchanged_facets':True, 'unchanged_lineage':True, 'index_strategy':'reuse parent sealed facet vectors and original audio query vectors; recompute supply against child snapshot', 'phase_supply':copy.deepcopy(base['knowledge_stores']['intent-boundaries']['phase_supply']), 'new_model_calls':0}
    write('patch.json', patch)
    write('audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(json.loads(Path(args.config).read_bytes())), ensure_ascii=False))


if __name__ == '__main__':
    main()
