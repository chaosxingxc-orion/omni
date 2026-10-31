"""Verify the real K4 plan against source-bound Solver receipts and requests.

Call after verify_bundle, before assigning trajectory credit. This check alone
is not a manifest, response, score or wire-transport verifier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
from speech_aware_evidence_acquisition.rails.trajectory_seeds import verify_recorded_group
import tfrl_v2_sampler as sampler


def verify(plan, output_rows, trace_records):
    columns=sampler.execution_columns(plan)
    expected={c['column_id']:c['decoding'] for c in columns}
    expected.update(aa=None,lock=None)
    ids=[row['sample_id'] for row in output_rows]
    if (len(ids)!=len(set(ids)) or set(ids)!={row['row_id'] for row in plan['row_plan']}):
        raise ValueError('sealed rows differ from the exact planned half')
    requests=[p['payload'] for rid,p in trace_records.items()
              if rid.endswith('/request') and isinstance(p.get('payload'),dict)]
    contacts=0
    for row in output_rows:
        receipt=row['supply_variants'];arms=receipt['arms']
        if receipt['schema']!='saea-supply-variant-arms-v6' or [a['arm_id'] for a in arms]!=list(expected):
            raise ValueError('sealed group differs from the seed-only plan')
        if any(a['decoding_override']!=expected[a['arm_id']] for a in arms):
            raise ValueError('sealed recipe differs from planned decoding')
        contacts+=verify_recorded_group(row['sample_id'],arms,requests)
    return {'schema':'saea-tfrl-seed-plan-request-check-v1','rows':len(ids),
        'solver_requests_verified':contacts,'new_model_calls':0,
        'scope':'Plan and request identity; requires independently verified bundle and official scores.'}


def audit(manifest,plan_path,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    if out.exists():raise ValueError('fresh audit output required')
    plan=json.loads(plan_path.read_bytes());bundle=verify_bundle(manifest)
    if (bundle.manifest['status']!='COMPLETED' or bundle.attempt_state!='FINALIZED'
            or bundle.manifest['run_id']!=plan['run_id']
            or bundle.manifest['attempt']['attempt_id']!=plan['attempt_id']):
        raise ValueError('finalized planned run and attempt required')
    result=verify(plan,bundle.outputs_rows,bundle.trace_records)
    result.update(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        run_id=plan['run_id'],attempt_id=plan['attempt_id'],
        raw_trace_sha256=bundle.manifest['raw_trace']['sha256'])
    out.parent.mkdir(parents=True,exist_ok=True)
    publish_new(out,(json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','plan_path','out'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    print(json.dumps(audit(**vars(p.parse_args()))))
