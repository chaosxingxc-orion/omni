"""Freeze reusable eligible TRAIN candidates from original labels and source custody.

No model/tokenizer/scorer contacts. Shared-P diagnostics nominate candidates,
not statistically confirmed hard examples. Existing audio/rosters are not copied.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from build_slurp_intent_exemplar_agreement import load_snapshot

ROSTERS = ('slurp-evo-fold-a-v2', 'slurp-evo-fold-b-v2', 'slurp-dev-500-b-v1',
           'slurp-dev-1000-ts2-v1', 'slurp-dev-2000-added-v1')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())
def uid(sid):
    parts = sid.split(':')
    if len(parts) < 3 or parts[0] != 'slurp' or not parts[2].isdigit():
        raise ValueError('original SLURP sample identity required')
    return ':'.join(parts[:3])
def text_key(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', text.casefold()).strip().encode()).hexdigest()


def coverage(rows):
    intents = Counter(r['intent'] for r in rows)
    slots = Counter(s for r in rows for s in set(r['slot_types']))
    relations = Counter((r['intent'], s) for r in rows for s in set(r['slot_types']))
    return {'utterances':len(rows), 'intents':len(intents), 'slot_types':len(slots),
            'observed_intent_slot_relations':len(relations),
            'intent_counts':dict(sorted(intents.items())),
            'slot_type_counts':dict(sorted(slots.items())),
            'relation_counts':[{'intent':i,'slot_type':s,'count':n} for (i,s),n in sorted(relations.items())]}


def partition(metadata, knowledge, evaluation, roster, evidence):
    """Select by immutable source roles first, then join already observed errors."""
    if len({r['slurp_id'] for r in metadata}) != len(metadata):
        raise ValueError('duplicate train utterance')
    rows = {f"slurp:train:{int(r['slurp_id']):06d}":r for r in metadata}
    blocked = set(knowledge) | set(evaluation)
    pool = set(rows) - blocked
    selected = {uid(r['sample_id']):r for r in roster}
    if len(selected) != len(roster): raise ValueError('roster duplicates utterance')
    by_sample = {r['row_id']:r for r in evidence}
    if len(by_sample) != len(evidence) or set(by_sample) != {r['sample_id'] for r in roster}:
        raise ValueError('evidence must cover existing roster exactly')
    blocked_text = {text_key(rows[u]['sentence']) for u in blocked if u in rows}
    records=[]; seen_text=set(); exclusion=Counter()
    for utterance in sorted(rows):
        raw=rows[utterance]; key=text_key(raw['sentence'])
        reason = ('knowledge_source' if utterance in knowledge else
                  'evaluation_source' if utterance in evaluation else
                  'duplicate_text_of_excluded_source' if key in blocked_text else
                  'duplicate_text_in_candidate_pool' if key in seen_text else None)
        if reason: exclusion[reason]+=1; continue
        seen_text.add(key)
        item=selected.get(utterance)
        sample=item['sample_id'] if item else None
        observation=by_sample.get(sample)
        candidate=bool(observation and observation['learning_candidate'])
        records.append({'utterance_id':utterance,'sample_id':sample,
            'intent':raw['scenario']+'_'+raw['action'],
            'slot_types':sorted(e['type'] for e in raw['entities']),
            'text_identity_sha256':key,'available_recordings':len(raw['recordings']),
            'audio_seconds':item['audio_seconds'] if item else None,
            'input_role':'fixed_learning_candidate_pool',
            'observation_role':'shared_perception_diagnostic' if observation else 'unobserved',
            'observed_error_candidate':candidate,'confirmed_hard_example':False,
            'full_system_repeat_count':0,
            'next_action':observation['next_action'] if observation else 'baseline_screen',
            'contrast_candidate_count':len(observation['contrast_candidates']) if observation else 0})
    return records, {'train_utterances':len(rows),'knowledge_source_utterances':len(set(knowledge)&set(rows)),
        'evaluation_source_utterances':len(set(evaluation)&set(rows)),
        'knowledge_evaluation_overlap':len(set(knowledge)&set(evaluation)&set(rows)),
        'role_eligible_before_text_dedup':len(pool),
        'role_eligible_equals_existing_roster':pool==set(selected),
        'excluded_by_primary_reason':dict(exclusion)}


def run(repo, source, data, credit, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output required')
    pins={}
    def pinned(path):
        pins[str(path.resolve().relative_to(data.resolve()))]=sha(path)
        return read(path)
    meta=data/'datasets/slurp/metadata/train.jsonl';pins[str(meta.relative_to(data))]=sha(meta)
    metadata=[json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    base=pinned(source/'configs/baseline/slurp-tfrl-current-seed-i1-h2-v1.json')
    knowledge=set();by_store={}
    for name,spec in sorted(base['knowledge_stores'].items()):
        snap,content_hash=load_snapshot(data,spec['snapshot'])
        pins[spec['snapshot']['relpath']]=sha(data/spec['snapshot']['relpath'])
        members={uid(s) for v in snap.values for s in v.source_sample_ids}
        knowledge |= members
        by_store[name]={'source_utterances':len(members),'content_sha256':content_hash}
    evaluation=set();evals={}
    for name in ROSTERS:
        path=source/'docs/receipts/rosters'/f'{name}.json';doc=pinned(path)
        members={uid(r['sample_id']) for r in doc['items']}
        evaluation|=members
        evals[name]={'rows':len(doc['items']),'utterances':len(members),'sha256':sha(path)}
    existing=source/'docs/receipts/rosters/slurp-stage3-construction-r01c.json'
    roster=pinned(existing)['items']
    completed=pinned(credit/'complete.json'); rowpath=credit/'rows.jsonl'
    if sha(rowpath)!=completed['rows_sha256'] or completed['rows']!=625:
        raise ValueError('credit row custody differs')
    evidence=[json.loads(line) for line in rowpath.read_text().splitlines()]
    if len(evidence)!=625: raise ValueError('complete625 evidence required')
    pins[str(rowpath.relative_to(data))]=sha(rowpath)
    records,summary=partition(metadata,knowledge,evaluation,roster,evidence)
    candidates=[r for r in records if r['observed_error_candidate']]
    all_cov=coverage(records);hard_cov=coverage(candidates)
    freq={(r['intent'],r['slot_type']):r['count'] for r in hard_cov['relation_counts']}
    priority=lambda r:(min([freq[(r['intent'],s)] for s in set(r['slot_types'])] or [10**9]),r['intent'],r['utterance_id'])
    queue=[{'sample_id':r['sample_id'],'utterance_id':r['utterance_id'],
            'job_kind':'independent_a0_difficulty_observation',
            'reason':r['next_action'],'status':'awaiting_registered_protocol',
            'model_calls_authorized':False,'observations_to_schedule':None}
           for r in sorted(candidates,key=priority)]
    out.mkdir(parents=True)
    for filename,items in [('candidate-pool.jsonl',records),('difficulty-observation-queue.jsonl',queue)]:
        publish_new(out/filename,b''.join((json.dumps(r,sort_keys=True)+'\n').encode() for r in items))
    train_cov=coverage([{'intent':r['scenario']+'_'+r['action'],'slot_types':[e['type'] for e in r['entities']]} for r in metadata])
    result={'schema':'saea-stable-hardset-candidate-manifest-v1',**summary,
        'implementation_sha256':sha(Path(__file__)),
        'source_files_sha256':pins,'knowledge_stores':by_store,'evaluation_rosters':evals,
        'train_coverage':train_cov,'eligible_pool_coverage':all_cov,'observed_error_coverage':hard_cov,
        'missing_candidate_intents':sorted(set(train_cov['intent_counts'])-set(hard_cov['intent_counts'])),
        'missing_candidate_slot_types':sorted(set(train_cov['slot_type_counts'])-set(hard_cov['slot_type_counts'])),
        'source_partition_expansion_requires_new_knowledge_version':summary['role_eligible_equals_existing_roster'],
        'candidate_actions':dict(Counter(r['next_action'] for r in candidates)),
        'pending_observation_rows':len(queue),'confirmed_hard_examples':0,
        'new_model_calls':0,'new_tokenizer_calls':0,'new_scoring_jobs':0,'test_labels_read':False,
        'artifacts':{f:sha(out/f) for f in ('candidate-pool.jsonl','difficulty-observation-queue.jsonl')},
        'limitations':['Existing shared-P controls are not independent full-system repeats.',
            'This freezes source-eligible candidates, not a statistically confirmed learning roster.',
            'The queue describes missing observations; no generation admission or call budget is implied.',
            'Observed intent-slot relations are coverage diagnostics, not a Cartesian product.',
            'Recorded train metadata contains no speaker identity; speaker disjointness is not verified.',
            'Duplicate-text isolation uses normalized train text; cross-split semantic equivalence is not proven.']}
    for rel,digest in pins.items():
        if sha(data/rel)!=digest: raise ValueError('source changed during read')
    publish_new(out/'complete.json',(json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
    return {k:result[k] for k in ('role_eligible_before_text_dedup','role_eligible_equals_existing_roster','pending_observation_rows','confirmed_hard_examples','candidate_actions')}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','source','data','credit','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),sort_keys=True))
