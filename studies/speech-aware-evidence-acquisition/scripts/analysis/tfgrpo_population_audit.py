"""CPU-only TRAIN coverage and H1/H2 identity audit; never select new samples."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(populations, labels, audio_hashes):
    if set(populations) != {'h1', 'h2', 'learned'}:
        raise ValueError('exact h1/h2/learned populations required')
    groups = {name: set(ids) for name, ids in populations.items()}
    if any(len(groups[k]) != len(v) for k, v in populations.items()):
        raise ValueError('duplicate sample identity')
    if not groups['learned'] <= groups['h1']:
        raise ValueError('learned rows must belong to H1')
    if not all(groups[k] <= labels.keys() and groups[k] <= audio_hashes.keys() for k in groups):
        raise ValueError('complete original labels and audio hashes required')
    def inventory(ids):
        counts = {'scenario': Counter(), 'intent': Counter(), 'slot': Counter(), 'intent_slot': Counter()}
        for sid in ids:
            row = labels[sid]
            counts['scenario'][row['scenario']] += 1
            counts['intent'][row['intent']] += 1
            for slot in set(row['slots']):
                counts['slot'][slot] += 1
                counts['intent_slot'][(row['intent'], slot)] += 1
        return counts
    all_train = inventory(labels)
    inventories = {k: inventory(v) for k, v in groups.items()}
    def overlap(a, b):
        return {'sample_ids': len(groups[a] & groups[b]),
                'utterance_ids': len({labels[s]['utterance_id'] for s in groups[a]} & {labels[s]['utterance_id'] for s in groups[b]}),
                'audio_hashes': len({audio_hashes[s] for s in groups[a]} & {audio_hashes[s] for s in groups[b]}),
                'normalized_train_sentences': len({labels[s]['sentence_sha256'] for s in groups[a]} & {labels[s]['sentence_sha256'] for s in groups[b]})}
    def encode(keys):
        return [list(k) if isinstance(k, tuple) else k for k in sorted(keys)]
    coverage = {}
    for name, inv in inventories.items():
        coverage[name] = {'rows': len(groups[name]), 'unique_utterances': len({labels[s]['utterance_id'] for s in groups[name]}),
            'original_train_coverage': {kind: {'covered': len(counts), 'total_observed_train': len(all_train[kind]),
                'missing': encode(all_train[kind].keys() - counts.keys()),
                'min_rows_per_present_label': min(counts.values(), default=0), 'max_rows_per_present_label': max(counts.values(), default=0)} for kind, counts in inv.items()}}
    return {'schema': 'saea-tfgrpo-population-audit-v1', 'populations': coverage,
            'h1_h2_overlap': overlap('h1', 'h2'), 'learned_h2_overlap': overlap('learned', 'h2'),
            'comparison_scope': 'Observed original TRAIN intent-slot relations only; no Cartesian product.',
            'limitations': ['Coverage is descriptive, not evidence of statistically frequent errors.',
                           'Audio identity uses pinned dataset hashes; this audit does not reread audio bytes.',
                           'Sentence overlap uses casefold and whitespace normalization only.',
                           'No sample selection, H2 error feedback or test labels used.'],
            'new_model_calls': 0, 'new_scoring_calls': 0}


def audit(repo, data, candidate):
    from speech_aware_evidence_acquisition.data.slurp import load_slurp_train_material
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    read = lambda p: json.loads(p.read_bytes())
    paths = {h: repo/f'docs/receipts/rosters/slurp-stage3-construction-r01c-{h}.json' for h in ('h1','h2')}
    view = repo/'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'
    rosters = {h: read(p) for h,p in paths.items()}
    labels = {}
    for row in load_slurp_train_material(load_lock(umbrella_lock_path(repo)), data):
        doc = {'utterance_id':f'slurp:train:{int(row.utterance_id):06d}', 'scenario':row.scenario,
               'intent':row.scenario+'_'+row.action, 'slots':list(row.entity_types),
               'sentence_sha256':hashlib.sha256(' '.join(row.sentence.casefold().split()).encode()).hexdigest()}
        for recording in row.recordings:
            if recording.sample_id in labels:raise ValueError('duplicate TRAIN recording')
            labels[recording.sample_id] = doc
    ids = {h:[r['sample_id'] for r in roster['items']] for h,roster in rosters.items()}
    ids['learned'] = read(candidate)['learned_query_ids']
    if (len(ids['h1']),len(ids['h2']),len(ids['learned'])) != (313,312,32):raise ValueError('fixed populations differ')
    for h,roster in rosters.items():
        for r in roster['items']:
            label=labels[r['sample_id']]
            if label['utterance_id']!=r['utterance_id'] or label['scenario']!=r['stratum']:raise ValueError('original roster identity differs')
    result=summarize(ids,labels,{r['sample_id']:r['audio_sha256'] for r in read(view)['items']})
    result['input_sha256']={**{h:hashlib.sha256(p.read_bytes()).hexdigest() for h,p in paths.items()},'dataset':hashlib.sha256(view.read_bytes()).hexdigest(),'candidate_file':hashlib.sha256(candidate.read_bytes()).hexdigest()}
    return result


if __name__ == '__main__':
    import argparse
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','candidate','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();assert_execution_environment()
    result=audit(a.repo,a.data,a.candidate)
    publish_new(a.out,json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True).encode()+b'\n')
    print(json.dumps({'populations':{k:v['rows'] for k,v in result['populations'].items()},'h1_h2_overlap':result['h1_h2_overlap'],'learned_h2_overlap':result['learned_h2_overlap']}))
