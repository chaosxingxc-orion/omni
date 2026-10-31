"""Audit candidate allocation against immutable real, context-intact requests.

No generation, labels, tokenizer or scorer. This is a source-shape prerequisite,
not a replay, FIT measurement or proof of a future banked request end to end.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
from tfrl_current_sources import read,sha,pinned
from speech_aware_evidence_acquisition.rails.tfrl_supply_allocation import declaration,keep_indices


def audit_rows(rows,trace,bound):
    allocation=declaration(bound);reports=[]
    for row in rows:
        arm=row['supply_variants']['arms'][-1]
        requests=[v['payload'] for k,v in trace.items() if k.endswith('/request')
            and v['payload'].get('sample_id')==row['sample_id']
            and (v['payload'].get('history') or [{}])[-1].get('contact_id')==arm['contact_id']]
        if len(requests)!=1:raise ValueError('one intact lock request required')
        payload=requests[0];items=payload['supplied_evidence']
        values=[x['evidence_id'] for x in items if x.get('knowledge_role')=='value']
        if values!=arm['supplied_value_ids']:raise ValueError('actual control render and custody order differ')
        indices=keep_indices(items,allocation);kept=[items[i] for i in indices]
        candidate=[x['evidence_id'] for x in kept if x.get('knowledge_role')=='value']
        if candidate!=values[:allocation['candidate_max_items']]:raise ValueError('actual candidate prefix differs')
        raw=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        reports.append({'source_request_sha256':hashlib.sha256(raw(payload)).hexdigest(),
            'control_value_items':len(values),'candidate_value_items':len(candidate),
            'removed_value_items':len(values)-len(candidate),'kept_nonvalue_items':len(kept)-len(candidate),
            'control_evidence_sha256':hashlib.sha256(raw(items)).hexdigest(),
            'candidate_evidence_sha256':hashlib.sha256(raw(kept)).hexdigest()})
    if not reports:raise ValueError('real intact requests required')
    return {'schema':'saea-tfrl-real-source-allocation-audit-v1','rows':len(reports),'allocation':allocation,
        'row_summaries':reports,'new_model_calls':0,'tokenizer_calls':0,'scoring_jobs':0,
        'labels_read':False,'scope':'Immutable context-intact original K4 lock request evidence; allocation only, no bank or FIT.',
        'remaining':['actual bank source bytes','full next-half shared rendering','final multimodal FIT','registered generation']}


def run(repo,source,data,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from tfrl_h1_remaining import sources
    from tfrl_h1_union import REL
    from tfrl_h1_recovery_custody import CHECKPOINT_SHA
    from tfrl_seed_request_audit import verify
    import stage3_rollout_score as s3
    assert_execution_environment()
    if not out.resolve().is_relative_to(data.resolve()/'work/codex') or out.exists():raise ValueError('fresh owned audit output required')
    source_head=require_clean_study_repo(source)
    recipe=read(source/REL);custody,parts=sources(source,data,recipe['custody'])
    checkpoint=pinned(data,custody['source_checkpoint']);cp=read_checkpoint(checkpoint,expected_sha256=CHECKPOINT_SHA)
    kept=set(parts['reusable_ids']);rows=[];trace={}
    for r in cp.rows:
        if r.sample_id not in kept:continue
        rows.extend(r.output_rows)
        for line in r.trace_lines:
            record=json.loads(line);key=record['record_id']
            if key in trace and trace[key]!=record['payload']:raise ValueError('trace namespace conflict')
            trace[key]=record['payload']
    plan=read(pinned(source,recipe['parent_plan']))
    verify({**plan,'row_plan':[r for r in plan['row_plan'] if r['row_id'] in kept]},rows,trace)
    base=read(source/'configs/baseline/slurp-tfrl-current-seed-i1-h1-v1.json')
    bound=base['native_episode']['consumption_module']['decoding_frame']['max_supplied_items']
    result=audit_rows(rows,trace,bound)
    result.update(source_commit=source_head,checkpoint_sha256=CHECKPOINT_SHA,custody_sha256=recipe['custody']['sha256'])
    s3._immutable_json(out/'complete.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','source','data','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args()))))
