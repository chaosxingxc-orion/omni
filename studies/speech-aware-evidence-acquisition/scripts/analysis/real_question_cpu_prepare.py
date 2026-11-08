"""Pinned, label-free question retrieval. Output is a reusable ranked cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot
from speech_aware_evidence_acquisition.rails.real_question_recall import validate_members


def checked(pin):
    p = Path(pin['path']); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    if h.hexdigest() != pin['sha256']: raise ValueError('source pin differs: ' + str(p))
    return p


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, sort_keys=True, separators=(',', ':')); f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(); p.add_argument('config'); p.add_argument('output'); args = p.parse_args()
    start = time.monotonic(); c = json.loads(Path(args.config).read_text())
    if c.get('aggregation') != 'max' or c['query_limit'] != c['expected_queries']: raise ValueError('invalid aggregation or query count')
    pool = json.loads(checked(c['pool']).read_bytes()); intent = json.loads(checked(c['intent_snapshot']).read_bytes())
    KnowledgeSnapshot.from_frozen_json(pool); KnowledgeSnapshot.from_frozen_json(intent)
    intents = {f['facet_id'].split(':', 1)[1]: f['facet_id'] for f in intent['facets'] if f['dimension'] == 'intent-boundary'}
    excluded = []
    for pin in c['rosters']:
        roster = json.loads(checked(pin).read_bytes()); excluded += [x.get('sample_id') or x.get('id') for x in roster['items']]
    query_ids = [x.get('sample_id') or x.get('id') for x in json.loads(checked(c['query_roster']).read_bytes())['items']]
    if len(query_ids) != c['expected_queries'] or len(set(query_ids)) != len(query_ids): raise ValueError('query roster count or duplicates')
    metadata = {}
    with checked(c['train']).open() as f:
        for line in f:
            row = json.loads(line); uid = str(row['slurp_id'])
            if uid in metadata: raise ValueError('duplicate TRAIN metadata id')
            metadata[uid] = row
    members = validate_members(pool, metadata, excluded, intents)
    if len(members) != c['expected_members'] or len({m[2] for m in members}) != c['expected_intents']: raise ValueError('unexpected pool size')
    required = {m[0] for m in members}; vectors = {}
    with checked(c['vectors']).open() as f:
        for line in f:
            row = json.loads(line)
            if row['id'] in required:
                if row['id'] in vectors: raise ValueError('duplicate vector')
                vectors[row['id']] = row['vectors']['content']
    if set(vectors) != required: raise ValueError('missing vector')
    queries = {}; wanted = set(query_ids)
    with checked(c['query']).open() as f:
        for line in f:
            row = json.loads(line)
            if row['id'] in wanted:
                if row['id'] in queries: raise ValueError('duplicate query')
                queries[row['id']] = [s['vectors']['intent-boundary'] for s in row['segments']]
    if set(queries) != wanted: raise ValueError('missing query vectors')
    load_seconds = time.monotonic() - start; t = time.monotonic()
    keys = np.asarray([vectors[m[0]] for m in members], dtype=np.float64)
    if keys.ndim != 2 or not np.isfinite(keys).all() or np.any(np.linalg.norm(keys,axis=1)==0): raise ValueError('invalid keys')
    keys /= np.linalg.norm(keys, axis=1)[:, None]
    groups = {fid: np.array([i for i,m in enumerate(members) if m[2] == fid]) for fid in sorted({m[2] for m in members})}
    # A chunk shares one matrix multiply and avoids a 1080x8554 permanent allocation.
    rows = {}; batch = c.get('batch_size', 64)
    for offset in range(0, len(query_ids), batch):
        ids = query_ids[offset:offset+batch]; segments = [np.asarray(queries[s], dtype=np.float64) for s in ids]
        if any(q.ndim != 2 or q.shape[1] != keys.shape[1] or not np.isfinite(q).all() or np.any(np.linalg.norm(q,axis=1)==0) for q in segments): raise ValueError('invalid query')
        allq = np.concatenate([q / np.linalg.norm(q,axis=1)[:,None] for q in segments]); sims = allq @ keys.T; pos = 0
        for sid,q in zip(ids, segments):
            scores = sims[pos:pos+len(q)].max(axis=0); pos += len(q)
            # Query identity is separate from source validation: only train IDs can self-match.
            import re
            match = re.fullmatch(r'slurp:train:([0-9]+):.+', sid)
            if match:
                uid = str(int(match.group(1)))
                for i,m in enumerate(members):
                    if m[1] == uid: scores[i] = -np.inf
            ranked = [(fid, float(scores[index].max())) for fid,index in groups.items()]
            rows[sid] = [[fid, score] for fid,score in sorted(ranked, key=lambda x: (-x[1],x[0])) if score > 0][:c['k']]
    # Multiple requested cutoffs are slices of the same complete, shared computation.
    for cutoff in c.get('additional_k', []):
        if type(cutoff) is not int or cutoff < 1 or cutoff > c['k']: raise ValueError('invalid additional_k')
    result = dict(schema='question-candidates-v1', rows=rows, source_config=c, samples=len(rows), pool_members=len(members), intent_count=len(groups), load_seconds=load_seconds, compute_seconds=time.monotonic()-t, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, model_calls=0, npu_cards=0)
    save(args.output, result)
    for cutoff in c.get('additional_k', []):
        variant = dict(result, rows={sid: ranking[:cutoff] for sid, ranking in rows.items()}, cutoff=cutoff)
        save(str(args.output) + '.k' + str(cutoff) + '.json', variant)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_config')}))

if __name__ == '__main__': main()
