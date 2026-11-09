"""Derive single mechanisms from frozen v5; never encode audio or use query labels."""
import argparse
import collections
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'delivery/s-tfrl-rolling-v1'
PARENT = ROOT / 'delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic'
sys.path.insert(0, str(ROOT / 'delivery/s-single-intent-fewshot-v1/code'))
from common import canonical, sha, read, lines, save, require, validate_packet
from retrieval import CachedRetrieval, example, features, select
from speech_aware_evidence_acquisition.rails import candidate_handoff as hb
from speech_aware_evidence_acquisition.rails import knowledge as kh
from speech_aware_evidence_acquisition.core.model.vllm_transport import vllm_request_body


def complement(pool, scores, records, groups, rule):
    prior, _, _ = select(pool, scores, records, groups)
    if len(prior) < 2:
        return prior, dict(reason='fewer_than_two_distinct_legal_sources')
    first = prior[0]
    remaining = sorted((j for j in pool if groups[j] != groups[first]), key=lambda j: (-float(scores[j]), j))
    best = float(scores[remaining[0]])
    eligible = [j for j in remaining[:rule['second_max_relevance_rank']]
                if float(scores[j]) >= best - rule['second_max_absolute_score_loss']]
    signature = collections.Counter(e['type'] for e in records[first]['entities'])
    first_types = set(signature)
    def novelty(j):
        sig = collections.Counter(e['type'] for e in records[j]['entities'])
        # Full observed TRAIN combinations, including a missing optional slot or empty row.
        return float(sig != signature), len(set(sig) ^ first_types) / max(1, len(set(sig) | first_types))
    def key(j):
        combo, optional = novelty(j)
        return (float(scores[j]) + rule['combination_bonus'] * combo + rule['optional_set_bonus'] * optional,
                float(scores[j]), -j)
    second = max(eligible, key=key)
    return [first, second], dict(first_column=first, original_second_column=prior[1], second_column=second,
        best_remaining_score=best, second_score=float(scores[second]), second_score_loss=best-float(scores[second]),
        second_relevance_rank=remaining.index(second)+1, relevance_eligible_columns=eligible,
        observed_combination_changed=bool(novelty(second)[0]), optional_set_distance=novelty(second)[1])


def timezone_snapshot():
    from prepare import project_rows
    original = kh.KnowledgeSnapshot.from_frozen_json(read(OUT.parent / 's-timezone-card-v1/prepared-v2/cards/snapshot.private.json'))
    authored = read(OUT.parent / 's-timezone-card-v1/prepared-v2/cards/authored.private.json')
    _, snapshot, projection = project_rows(authored, original)
    parent = read(PARENT / 'snapshot.projected.private.json')
    changed = []
    for before, after in zip(parent['values'], snapshot.frozen_json()['values']):
        require(before['value_id'] == after['value_id'], 'projected roster')
        if before != after:
            require(before['value_id'].endswith('::slot-contract:time_zone'), 'other projected card changed')
            changed.append(before['value_id'])
    require(len(changed) == 3, 'three timezone bindings only')
    return snapshot, projection, changed


def main(kind):
    start, cpu = time.monotonic(), time.process_time()
    name = 'timezone-single-v1' if kind == 'timezone' else 'fewshot-complement-v1'
    output = OUT / 'candidates' / name
    require(not output.exists(), 'immutable fresh candidate only')
    arm = 'single_intent_no_examples' if kind == 'timezone' else 'single_intent_fewshot'
    prior_cfg = read(PARENT / arm / 'native-UNBOUND.json')
    parent_manifest = PARENT / arm / 'manifest.private.jsonl'
    require(sha(parent_manifest.read_bytes()) == prior_cfg['manifest_sha256'], 'v5 parent manifest pin')
    old_snapshot = read(PARENT / 'snapshot.projected.private.json')
    if kind == 'timezone':
        snapshot, projection, changed_values = timezone_snapshot()
        save(output / 'projection.private.json', projection)
    else:
        snapshot = kh.KnowledgeSnapshot.from_frozen_json(old_snapshot)
        changed_values = []
    snap = snapshot.content_hash()
    values = {v.value_id: v for v in snapshot.values}
    save(output / 'snapshot.projected.private.json', snapshot.frozen_json())
    save(output / 'samples.private.jsonl', lines(PARENT / 'samples.private.jsonl'), jsonl=True)
    retrieval = None
    if kind == 'fewshot':
        roster = lines(ROOT / read(ROOT / 'delivery/s545-baseline-lock-v1/LOCK.json')['manifest'])
        retrieval = CachedRetrieval(ROOT / 'delivery/s-single-intent-fewshot-v1/config/retrieval-v5.json', roster)
        rule = read(OUT / 'config/complement-v1.json')
        save(output / 'selection-rule.json', rule)
        save(output / 'source-policy.json', retrieval.policy | dict(selection='relevance_constrained_observed_combination_complement_v1', query_gold_used=False))
    reports, manifest, changed_packets, example_changes = [], [], 0, 0
    for prior in lines(parent_manifest):
        raw = (ROOT / prior['payload']).read_bytes()
        require(sha(raw) == prior['packet_sha256'], 'parent packet custody')
        packet = json.loads(raw)
        before = copy.deepcopy(packet)
        cid = prior['candidate_ids'][0]
        blocks = packet['wire']['messages'][1]['content']
        if kind == 'timezone':
            for part in blocks:
                if part.get('knowledge_role') == 'value':
                    part['text'] = values[part['evidence_id']].content
                    part['source_uri'] = 'knowledge-snapshot://' + snap + '/' + part['evidence_id']
            h = packet['handoff']
            knowledge = []
            for group in h['knowledge']:
                for vid in group['ids']:
                    value = values[vid]
                    knowledge.append(dict(id=vid, snapshot_sha256=snap, text=value.content,
                        scope=dict(facets=list(value.associated_facet_ids), sources=list(value.source_sample_ids),
                                   provenance=list(value.provenance_refs)), annotations=group['annotations'], target_role='slot'))
            packet['handoff'] = hb.build_handoff(sample_id=h['sample_id'], source_role=h['source_role'], target_role=h['target_role'],
                source_response_sha256=h['source_response_sha256'], snapshot_sha256=snap,
                candidate_ids=h['candidate_ids'], judgments=h['judgments'], knowledge=knowledge)
            # Derived inventory hashes/URIs are identity fields, not extra instructions.
            text = blocks[-1]['text'].replace(old_snapshot['snapshot_sha256'], snap)
            for vid in changed_values:
                old = next(v for v in old_snapshot['values'] if v['value_id'] == vid)
                text = text.replace(sha(old['content'].encode()), sha(values[vid].content.encode()))
            blocks[-1]['text'] = text
            require(packet['wire']['messages'][0] == before['wire']['messages'][0], 'system instruction changed')
        else:
            query = retrieval.rows[prior['source_sample_id']]
            scores = retrieval.scores[query['score_row']]
            pool = [j for j in retrieval.pools[cid] if retrieval.groups[j] not in set(query['excluded_sourcegroups'])
                    and {e['type'] for e in retrieval.records[j]['entities']} <= set(packet['allowed_types'])]
            chosen, decision = complement(pool, scores, retrieval.records, retrieval.groups, rule)
            examples = [example(retrieval.records[j], cid) for j in chosen]
            require(not examples or examples[0] == before['examples'][0], 'first relevance neighbor changed')
            if packet['examples']:
                require(blocks[-2]['text'].startswith('TRAIN demonstrations (not current-audio evidence):\n'), 'demo block')
                blocks.pop(-2)
            packet['examples'] = examples
            if examples:
                demo = [{k: e[k] for k in ['candidate_id','input','annotation','output','annotated_mentions']} for e in examples]
                blocks.insert(-1, dict(type='text', text='TRAIN demonstrations (not current-audio evidence):\n' + canonical(demo).decode()))
            require(packet['handoff'] == before['handoff'], 'fewshot card/handoff changed')
            example_changes += examples != before['examples']
            reports.append(dict(branch_key=packet['sample_id'], sample_id=prior['source_sample_id'], candidate_id=cid,
                score_row=query['score_row'], selected_columns=chosen, selected_scores=[float(scores[j]) for j in chosen],
                selected_source_groups=[retrieval.groups[j] for j in chosen], legal_pool_sha256=sha(canonical(pool)),
                first_unchanged=True, **decision))
        require(packet['wire']['messages'][1]['content'][0] == before['wire']['messages'][1]['content'][0], 'audio mutated')
        require(packet['wire']['response_format'] == before['wire']['response_format'], 'schema mutated')
        validate_packet(packet, selected=[cid], expected_audio=before['wire']['messages'][1]['content'][0], snapshot=snap,
            values={k:v.content for k,v in values.items()}, handoff_module=hb)
        changed_packets += packet != before
        path = output / arm / 'packets' / Path(prior['payload']).name
        save(path, packet)
        row = copy.deepcopy(prior)
        row.update(payload=str(path.relative_to(ROOT)), packet_sha256=sha(path.read_bytes()),
            abstract_wire_sha256=sha(canonical(packet['wire'])),
            wire_sha256=sha(canonical(vllm_request_body(packet['wire'], packet['wire']['model']))),
            handoff_sha256=packet['handoff']['handoff_sha256'])
        manifest.append(row)
    require(len(manifest) == 1635 and len({r['source_sample_id'] for r in manifest}) == 545, 'fixed545x3')
    path = output / arm / 'manifest.private.jsonl'
    save(path, manifest, jsonl=True)
    cfg = copy.deepcopy(prior_cfg)
    cfg.update(run_id='S-ROLLING-' + name.upper(), attempt_id='v1', manifest=str(path.relative_to(ROOT)),
        manifest_sha256=sha(path.read_bytes()), snapshot=str((output/'snapshot.projected.private.json').relative_to(ROOT)),
        snapshot_sha256=snap, prepared=str(output.relative_to(ROOT)), samples=str((output/'samples.private.jsonl').relative_to(ROOT)))
    save(output / arm / 'native-UNBOUND.json', cfg)
    save(output / 'selection-audit.private.jsonl', reports, jsonl=True)
    save(output / 'DIFF.json', dict(kind=kind, parent=str(parent_manifest.relative_to(ROOT)), parent_sha256=prior_cfg['manifest_sha256'],
        changed_packets=changed_packets, changed_example_branches=example_changes, changed_projected_values=changed_values,
        samples=545, branches=1635, query_gold_used=False, audio_encodes=0, model_calls=0))
    save(output / 'CPU-PREPARED.json', dict(status='INPUTS_PREPARED_NEEDS_FIXED_IMAGE_BUDGET', samples=545, branches=1635,
        cards=0, instances=0, wall_seconds=time.monotonic()-start, CPU_seconds=time.process_time()-cpu,
        seconds_per_sample=(time.monotonic()-start)/545, inference_completed=0, inference_total=1635,
        inference_ETA_seconds=None, money=None, parent_version='prepared-v5-dynamic', changes=read(output/'DIFF.json')))
    print(json.dumps(read(output/'CPU-PREPARED.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['timezone', 'fewshot'])
    main(p.parse_args().kind)
