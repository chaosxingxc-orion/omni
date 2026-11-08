"""Deterministic, label-local selection of original TRAIN question members."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FIELDS = {'min_per_label', 'scale', 'cap_per_label', 'boundary_fraction',
          'margin_weight', 'crosshub_weight', 'redundancy_weight'}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('exact selection configuration required')
    for key, values in [('min_per_label', (8, 16, 32)), ('scale', (2, 4, 6)),
                        ('cap_per_label', (64, 128, 256)), ('boundary_fraction', (0, .1, .2))]:
        if type(config[key]) not in (int, float) or config[key] not in values:
            raise ValueError('invalid '+key)
    for key in ('margin_weight', 'crosshub_weight', 'redundancy_weight'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError('invalid '+key)
    return config


def midrank(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([.5])
    order = np.argsort(values, kind='stable'); result = np.empty(len(values)); i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = ((i + j - 1) / 2) / (len(values) - 1)
        i = j
    return result


def crosshub(topidx, labels):
    labels = np.asarray(labels); _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    neighbors = np.asarray(topidx); out = np.zeros(len(labels))
    for i, row in enumerate(neighbors):
        for j in row:
            if j >= 0 and labels[j] != labels[i]:
                out[j] += 1.0 / counts[inverse[i]]
    return out


def greedy(similarity, representatives, margin, hub, nearest_same, config):
    """Select representatives while covering every original member; columns index representatives."""
    validate_config(config)
    matrix = np.clip(np.asarray(similarity, dtype=float), 0, 1)
    reps = np.asarray(representatives, dtype=int); n, k = matrix.shape
    if k != len(reps) or not k or len(set(reps.tolist())) != k:
        raise ValueError('nonempty distinct representatives required')
    quality = midrank(margin); penalty = midrank(hub)
    if len(quality) != n or len(penalty) != n or len(nearest_same) != n:
        raise ValueError('original class geometry dimensions differ')
    budget = min(k, max(config['min_per_label'], min(config['cap_per_label'], math.ceil(config['scale'] * math.sqrt(k)))))
    eligible = (quality[reps] <= .25) & (np.asarray(nearest_same)[reps] >= np.median(nearest_same))
    reserve = math.ceil(budget * config['boundary_fraction']); chosen = []; audit = []
    coverage = np.zeros(n); redundancy = np.zeros(k); available = np.ones(k, dtype=bool)
    for step in range(budget):
        boundary = step < reserve and np.any(eligible & available)
        pool = np.flatnonzero(available & eligible) if boundary else np.flatnonzero(available)
        # Gains normalize across all remaining candidates, including outside the boundary pool.
        remaining = np.flatnonzero(available)
        gains = np.maximum(matrix[:, remaining] - coverage[:, None], 0).mean(axis=0)
        lo, hi = gains.min(), gains.max(); normalized = np.zeros_like(gains) if hi == lo else (gains-lo)/(hi-lo)
        scores = normalized + config['margin_weight']*quality[reps[remaining]] - config['crosshub_weight']*penalty[reps[remaining]] - config['redundancy_weight']*redundancy[remaining]
        lookup = {int(v): i for i, v in enumerate(remaining)}
        best = min(pool, key=lambda j: (-scores[lookup[int(j)]], int(reps[j])))
        x = lookup[int(best)]; chosen.append(int(best)); audit.append({'step':step+1,'original_index':int(reps[best]),'boundary_slot':bool(boundary),'gain':float(gains[x]),'normalized_gain':float(normalized[x]),'score':float(scores[x]),'margin_quality':float(quality[reps[best]]),'hub_penalty':float(penalty[reps[best]]),'redundancy':float(redundancy[best])})
        coverage = np.maximum(coverage, matrix[:, best]); redundancy = np.maximum(redundancy, matrix[reps, best]); available[best] = False
    closest = np.argmax(matrix[:, chosen], axis=1)
    return chosen, audit, coverage, closest


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def cache_identity(paths):
    return {'schema':'question-selection-similarity-v1','inputs':[digest(p) for p in paths],'algorithm':'clip-cosine-original-by-representatives-v1'}


def group_members(members, intents):
    grouped = {}
    for intent in intents:
        originals = [i for i, m in enumerate(members) if m['intent'] == intent]
        seen = set(); reps = []
        for local, i in enumerate(originals):
            key = members[i]['sentence_sha256']
            if key not in seen:
                seen.add(key); reps.append(local)
        grouped[intent] = (np.array(originals, dtype=int), np.array(reps, dtype=int))
    return grouped


def prepare_cache(keys_path, members_path, geometry_path, rankings_path, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    identity = cache_identity([keys_path, members_path, geometry_path, rankings_path])
    members = json.loads(Path(members_path).read_bytes()); intents = json.loads(Path(rankings_path).read_bytes())['intents']
    keys = np.load(keys_path, mmap_mode='r'); geometry = np.load(geometry_path); labels = [m['intent'] for m in members]
    if keys.shape[0] != len(members) or geometry['topidx'].shape[0] != len(members): raise ValueError('source dimensions differ')
    hubs = crosshub(geometry['topidx'], labels)
    for index, intent in enumerate(intents):
        original, reps = group_members(members, [intent])[intent]
        if len(original):
            sim = np.clip(keys[original] @ keys[original[reps]].T, 0, 1)
            np.savez_compressed(out / f'{index:02d}.npz', original=original, reps=reps, similarity=sim, hub=hubs[original])
    (out/'identity.json').write_text(json.dumps(identity, sort_keys=True, indent=2)+'\n')
    return identity


def select(cache, keys, members_path, geometry_path, rankings_path, source_geometry_path, config, out):
    validate_config(config); cache = Path(cache); expected = cache_identity([keys,members_path,geometry_path,rankings_path])
    if json.loads((cache/'identity.json').read_bytes()) != expected: raise ValueError('cached source identity differs')
    members = json.loads(Path(members_path).read_bytes()); source = json.loads(Path(source_geometry_path).read_bytes()); intents = json.loads(Path(rankings_path).read_bytes())['intents']; geometry = np.load(geometry_path)
    if len(source) != len(members) or any(x['member_index'] != i or x['intent'] != members[i]['intent'] or x['sentence_sha256'] != members[i]['sentence_sha256'] for i,x in enumerate(source)): raise ValueError('source geometry identity differs')
    if not np.array_equal(np.asarray([x['mean5_margin'] for x in source]), geometry['positive']-geometry['negative']) or not np.array_equal(np.asarray([x['nearest_same'] for x in source]), geometry['same_max']): raise ValueError('source geometry values differ from pinned geometry')
    chosen_all=[]; selected=[]; omitted=[]; classes={}
    for index, intent in enumerate(intents):
        path = cache/f'{index:02d}.npz'
        if not path.exists(): classes[intent]={'original':0,'unique':0,'retained':0}; continue
        data=np.load(path); orig=data['original']; reps=data['reps']; sim=data['similarity']; margin=[source[i]['mean5_margin'] for i in orig]; near=[source[i]['nearest_same'] for i in orig]
        positions,audit,coverage,closest=greedy(sim,reps,margin,data['hub'],near,config); selected_local=reps[positions]; chosen_all.extend(orig[selected_local].tolist());
        for row in audit:
            local=row.pop('original_index'); idx=int(orig[local]);selected.append({**row,'member_index':idx,'intent':intent,'source':members[idx]})
        chosen_local_set=set(selected_local.tolist())
        for local, idx in enumerate(orig):
            if local not in chosen_local_set:
                nearest=int(orig[selected_local[closest[local]]]); omitted.append({'member_index':int(idx),'intent':intent,'nearest_retained_index':nearest,'coverage_similarity':float(coverage[local]),'coverage_gap':float(1-coverage[local])})
        classes[intent]={'original':len(orig),'unique':len(reps),'retained':len(positions),'boundary_selected':sum(a['boundary_slot'] for a in audit)}
    out=Path(out);out.mkdir(parents=True,exist_ok=False);mask=np.zeros(len(members),dtype=bool);mask[chosen_all]=True;np.save(out/'mask.npy',mask)
    for name,value in [('selected.json',selected),('omitted.json',omitted),('classes.json',classes),('config.json',config)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
    return {'selected':len(chosen_all),'original':len(members),'classes':classes,'cache_identity':expected}
