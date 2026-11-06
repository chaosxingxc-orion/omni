#!/usr/bin/env python3
"""Validate or publish retrospective not-sent proof; never send or settle lease."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, digest
from speech_aware_evidence_acquisition.core.contact_reconciliation import SCHEMA, POLICY, validate, _read


def execute(config, *, publish=False):
    if set(config) != {'schema','data','plan','arm','request_ids','evidence','lock_directory'} or config['schema'] != 'saea-pre-dispatch-reconciliation-config-v1':
        raise ValueError('exact reconciliation config required')
    plan_doc = json.loads(_read(config['plan']))
    plan = ExecutionPlan(**plan_doc['execution_plan'])
    journal = ContactJournal(Path(config['data']), plan, config['arm'])
    ids = config['request_ids']
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('unique explicit request IDs required')
    def prepare():
        results = []
        for rid in ids:
            intent_path = journal._path(rid, 'intent')
            blob = intent_path.read_bytes(); intent = json.loads(blob)
            state = journal.inspect(rid)['state']
            if state != 'unknown' or journal._path(rid, 'reconciliation').exists():
                raise ValueError('only an unresolved legacy request may be reconciled')
            if any(journal._path(rid, k).exists() for k in ('response','dispatch','not_sent')):
                raise ValueError('live dispatch evidence cannot be retrospectively replaced')
            doc = {'schema':SCHEMA,'policy':POLICY,'identity':journal.identity,
                   'request_id':rid,'request_sha256':intent['request_sha256'],
                   'intent_sha256':digest(blob),'verdict':'not_sent','evidence':config['evidence']}
            proof = validate(doc, intent_blob=blob, identity=journal.identity)
            results.append((doc, proof))
        return results
    if not publish:
        return {'published':False,'proofs':[r for _, r in prepare()],'new_model_calls':0}
    import subprocess
    from tfgrpo_recovery_queue import exclusive
    locks = Path(config['lock_directory'])
    with exclusive(locks/'generation.guard'):
        lease_blob = _read(config['evidence']['lease']); lease = json.loads(lease_blob)
        if (locks/'flight.lock').read_bytes() != lease_blob:
            raise ValueError('live lease differs; no reconciliation publication')
        expected = json.loads(_read(config['evidence']['container']))[0]
        current, = json.loads(subprocess.check_output(['docker','inspect',expected['Id']]))
        if current['State'] != expected['State'] or current['State']['Running'] or Path('/proc',str(lease['pid'])).exists():
            raise ValueError('exact producer is not verified stopped')
        attempt_path = Path(config['data'])/'attempts'/plan.run_id/(plan.attempt_id+'.json')
        if attempt_path.read_bytes() != _read(config['evidence']['attempt']):
            raise ValueError('live attempt differs from pinned evidence')
        results = prepare()
        for doc, _ in results:
            publish_new(journal._path(doc['request_id'], 'reconciliation'), canonical(doc))
    return {'published':True,'proofs':[r for _,r in results], 'new_model_calls':0,'lease_modified':False}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--publish',action='store_true')
    args=p.parse_args()
    print(json.dumps(execute(json.loads(args.config.read_bytes()),publish=args.publish),sort_keys=True))
