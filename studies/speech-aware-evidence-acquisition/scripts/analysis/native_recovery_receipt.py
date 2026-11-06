"""Publish a verified operational recovery receipt for an explicitly aborted run.

This command never aborts a live attempt or sends model requests. Unknown
contacts remain unknown; the runtime refuses to resend them automatically.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.contact_journal import (
    NativeContactReplay, ContactJournal, canonical, digest,
)
from speech_aware_evidence_acquisition.core.flight.native_recovery import load_native_recovery


def prepare(data_root, source_plan, target_attempt, arm, out):
    """Validate durable custody and publish once; preserve scientific parameters."""
    data_root = Path(data_root).resolve()
    out = Path(out)
    if not out.is_absolute() or not out.resolve().is_relative_to(data_root):
        raise ValueError('absolute output confined to data root required')
    source_plan.validate()
    target = replace(source_plan, attempt_id=target_attempt).validate()
    journal = ContactJournal(data_root, target, arm)
    existing = (data_root/'attempts'/target.run_id/(target.attempt_id+'.json'),
                data_root/'runs'/target.run_id/target.attempt_id, journal.root)
    if any(p.exists() for p in existing):
        raise ValueError('target attempt already has state; inspect instead of overwriting')
    replay = NativeContactReplay(data_root, source_plan, target, arm)
    counts = {'observed': 0, 'unknown': 0}
    if any(replay.source.inspect(rid)['state'] == 'not_sent' for rid in replay.request_ids):
        counts['not_sent'] = 0
    for rid in sorted(replay.request_ids):
        state = replay.source.inspect(rid)['state']
        counts[state] += 1
    target_document = asdict(target)
    if target.owner_allowance is not None:
        target_document['owner_allowance'] = asdict(target.owner_allowance)
    document = {'schema': 'saea-native-recovery-v1',
                'source': replay.source_binding(), 'target_plan': target_document}
    blob = canonical(document)
    out.parent.mkdir(parents=True, exist_ok=True)
    publish_new(out, blob)
    declaration = {'path': str(out), 'sha256': digest(blob)}
    # Use the same loader as worker preflight; a receipt is not execution authority.
    load_native_recovery(declaration, data_root=data_root, plan=target, arm=arm)
    return {'declaration': declaration, 'source_attempt': source_plan.attempt_id,
            'target_attempt': target.attempt_id, 'contacts': counts,
            'generation_started': False, 'unknown_auto_retry': False,
            'source_liveness': 'requires prior explicit ABORTED custody; not changed here'}


if __name__ == '__main__':
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source-plan', type=Path, required=True,
                        help='Exact registered ExecutionPlan JSON; never inferred from partial logs')
    parser.add_argument('--target-attempt', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = vars(parser.parse_args())
    args['source_plan'] = ExecutionPlan(**json.loads(args['source_plan'].read_bytes()))
    print(json.dumps(prepare(**args), sort_keys=True))
