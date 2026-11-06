"""Read-only audit of existing populations; no sampling or inference."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
paths = {
    'fold_a': 'configs/dataset/slurp-speech-native-evo-fold-a-1080.json',
    'fold_b': 'configs/dataset/slurp-speech-native-evo-fold-b-1080.json',
    'construction': 'configs/dataset/slurp-speech-native-stage3-construction-r01c.json',
    'construction_roster': 'docs/receipts/rosters/slurp-stage3-construction-r01c.json',
    'h1': 'docs/receipts/rosters/slurp-stage3-construction-r01c-h1.json',
    'h2': 'docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json',
}
sha = lambda data: hashlib.sha256(data).hexdigest()
canonical = lambda data: json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
docs = {key: json.loads((root / rel).read_bytes()) for key, rel in paths.items()}
sets = {}
summaries = {}
for key, doc in docs.items():
    rows = doc['items']
    ids = [r['sample_id'] for r in rows]
    assert len(ids) == len(set(ids)), (key, 'duplicate sample')
    assert all(i.startswith('slurp:train:') and len(i.split(':', 3)) == 4 for i in ids)
    utterances = {':'.join(i.split(':', 3)[:3]) for i in ids}
    assert all(r.get('utterance_id', ':'.join(r['sample_id'].split(':', 3)[:3])) == ':'.join(r['sample_id'].split(':', 3)[:3]) for r in rows)
    assert doc['split_identity_hash'] == sha('\n'.join(sorted('slurp/' + i for i in ids)).encode())
    if 'runtime_view_sha256' in doc:
        assert doc['runtime_view_sha256'] == sha(canonical({k:v for k,v in doc.items() if k != 'runtime_view_sha256'}))
        assert doc['count'] == len(rows)
    audio = {r['audio_sha256'] for r in rows if 'audio_sha256' in r}
    sets[key] = {'samples': set(ids), 'utterances': utterances, 'audio': audio}
    summaries[key] = {'path': paths[key], 'sha256': sha((root / paths[key]).read_bytes()),
        'samples': len(ids), 'utterances': len(utterances), 'audio_digests': len(audio),
        'split_identity_verified': True, 'runtime_view_verified': 'runtime_view_sha256' in doc}
assert sets['construction']['samples'] == sets['construction_roster']['samples']
assert sets['construction']['samples'] == sets['h1']['samples'] | sets['h2']['samples']
media_rel = 'docs/' + docs['construction']['d1_media_supplement']
media_bytes = (root / media_rel).read_bytes()
assert sha(media_bytes) == docs['construction']['d1_media_supplement_sha256']
assert sha(media_bytes) == docs['construction_roster']['d1_supplement']['file_sha256']
media = json.loads(media_bytes)
records = {r['id']:r for r in media['manifest']}
assert len(records) == len(media['manifest']) == media['count']
for r in docs['construction']['items']:
    m = records[r['sample_id']]
    assert (r['audio_sha256'],r['audio_seconds'],r['sample_rate_hz'],r['media_relpath']) == (m['sha256'],m['audio_seconds'],m['sample_rate_hz'],m['media_relpath'])
index = {r['sample_id']: r for r in docs['construction']['items']}
for key in ('h1','h2'):
    sets[key]['audio'] = {index[i]['audio_sha256'] for i in sets[key]['samples']}
    summaries[key]['audio_digests_resolved_from_parent'] = len(sets[key]['audio'])
comparisons = []
for a,b in [('fold_a','fold_b'),('construction','fold_a'),('construction','fold_b'),('h1','h2')]:
    overlap = {key: len(sets[a][key] & sets[b][key]) for key in ('samples','utterances','audio')}
    comparisons.append({'left':a,'right':b,'overlap':overlap})
result = {'schema':'omni-psd-training-population-audit-v1', 'inputs':summaries,
    'comparisons':comparisons, 'all_examined_intersections_empty':all(not any(x['overlap'].values()) for x in comparisons),
    'h1_h2_exact_partition':True,'construction_media_manifest':{'path':media_rel,'sha256':sha(media_bytes),'parent_records':len(records),'matched_records':len(index)},
    'scope':'Existing roster/view/manifest identity and overlap only; no raw audio bytes, annotation labels, ancestor exposure, current rollout availability or model performance verified.',
    'new_sample_ids':0,'model_calls':0,'resampling':False}
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'comparisons':comparisons,'all_empty':result['all_examined_intersections_empty']}))
