"""Verify the existing minus-R01 pool before current-port TFRL preparation.

No query labels, audio, encoder or inference contacts are used. Compare complete
utterance identities, frozen source membership and shared-key vector coverage.
"""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
from fold_a_retrieval_inputs import read,sha,raw,confined
from r01c_encoding_write_registration import ROSTER
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_lock_v2_2_npu_port_r01c import R01C_RECEIPT_RELPATH


def utterance(sid):
    fields=sid.split(':')
    if len(fields)<3 or fields[:2]!=['slurp','train'] or not fields[2]:
        raise ValueError('expected SLURP train utterance identity')
    return ':'.join(fields[:3])


def verify_membership(query_ids, snapshot):
    if len(query_ids)!=625 or len(set(query_ids))!=625:
        raise ValueError('complete unique R01C625 required')
    query={utterance(sid) for sid in query_ids}
    source=set(snapshot.source_sample_ids)
    if not source or len(source)!=len(snapshot.source_sample_ids):
        raise ValueError('empty or duplicate source membership')
    if query & {utterance(sid) for sid in source}:
        raise ValueError('pool source overlaps query utterances')
    for item in (*snapshot.facets,*snapshot.values):
        members=set(item.source_sample_ids)
        if not members or not members<=source or query & {utterance(sid) for sid in members}:
            raise ValueError('facet/value source escapes disjoint pool membership')
    facet_ids={f.facet_id for f in snapshot.facets}
    if len(facet_ids)!=len(snapshot.facets):raise ValueError('duplicate pool facets')
    for value in snapshot.values:
        if not set(value.associated_facet_ids)<=facet_ids or not value.task_target:
            raise ValueError('pool value has missing facet or vote target')
    return {'query_rows':len(query_ids),'query_utterances':len(query),
        'pool_source_recordings':len(source),'pool_utterances':len({utterance(s) for s in source}),
        'facets':len(snapshot.facets),'values':len(snapshot.values),'utterance_overlap':0}


def audit(repo,data,source_baseline,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot
    from speech_aware_evidence_acquisition.rails.phase_supply import load_sealed_vectors
    assert_execution_environment()
    receipt_path=repo/R01C_RECEIPT_RELPATH;receipt=read(receipt_path)
    snapshot_path=confined(data,receipt['snapshot_artifact_relpath'])
    release_path=confined(data,receipt['release_artifact_relpath'])
    if (sha(snapshot_path)!=receipt['snapshot_file_sha256']
            or sha(release_path)!=receipt['release_file_sha256']):
        raise ValueError('frozen minus-R01 store bytes differ')
    snapshot=KnowledgeSnapshot.from_frozen_json(read(snapshot_path));release=read(release_path)
    if (snapshot.content_hash()!=receipt['snapshot_sha256']
            or release['snapshot_sha256']!=snapshot.content_hash()
            or release['source_split_identity_hash']!=snapshot.source_split_identity_hash):
        raise ValueError('release and snapshot content identity differ')
    rows=read(repo/ROSTER)['items'];counts=verify_membership([r['sample_id'] for r in rows],snapshot)
    baseline=read(source_baseline)
    spec=baseline['knowledge_stores']['train-exemplars']['phase_supply']['solver']
    pin=spec['facet_source'];vector_path=confined(data,pin['relpath'])
    vectors=load_sealed_vectors(vector_path,expected_sha256=pin['sha256'],dimensions=('content',))
    facet_ids={f.facet_id for f in snapshot.facets}
    if not facet_ids<=set(vectors):raise ValueError('missing pool facet vectors')
    if any(set(vectors[f])!={'content'} or len(vectors[f]['content'])!=2048 for f in facet_ids):
        raise ValueError('pool facet vector geometry differs')
    if (counts['facets']!=receipt['facet_count'] or counts['values']!=receipt['value_count']
            or snapshot.historical_prior_score_range_violations):
        raise ValueError('pool counts or prior probabilities differ')
    out=confined(data,str(out))
    if not out.is_relative_to((data/'work/codex').resolve()):raise ValueError('owned audit output required')
    record={'schema':'saea-r01c-current-pool-audit-v1',**counts,
        'vector_rows':len(vectors),'covered_pool_facets':len(facet_ids),'vector_width':2048,
        'pins':{'pool_receipt':sha(receipt_path),'roster':sha(repo/ROSTER),
            'snapshot_file':sha(snapshot_path),'release':sha(release_path),
            'facet_vectors':sha(vector_path),'source_baseline':sha(source_baseline)},
        'snapshot_content_sha256':snapshot.content_hash(),
        'recall':{k:spec[k] for k in ('k','rank_policy','rank_weights','fuse_policy','fuse_coverage')},
        'new_model_contacts':0,'new_encoder_contacts':0,'query_vectors_used':False,
        'claims_agreement_ready':False,'claims_tfrl_admitted':False}
    publish_new(out,raw(record));return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','source-baseline','out'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(audit(**vars(p.parse_args()))))
