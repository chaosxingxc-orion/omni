#!/usr/bin/env python3
"""Reconcile a completed score whose final attempt lock failed; never rescore."""
import argparse
import fcntl
import hashlib
import json
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.attempts import AttemptHandle
from speech_aware_evidence_acquisition.core.bundle import verify_bundle
from speech_aware_evidence_acquisition.core.tracking import ledger_row_for_bundle


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def terminal(cid, socket_path):
    require(len(cid) == 64 and all(c in '0123456789abcdef' for c in cid), 'invalid container id')
    sock = socket.socket(socket.AF_UNIX)
    sock.settimeout(10)
    try:
        sock.connect(socket_path)
        sock.sendall(f'GET /containers/{cid}/json HTTP/1.0\r\nHost: localhost\r\n\r\n'.encode())
        data = b''
        while True:
            part = sock.recv(65536)
            if not part:
                break
            data += part
    finally:
        sock.close()
    header, body = data.split(b'\r\n\r\n', 1)
    require(header.splitlines()[0].split()[1] == b'200', 'container inspection failed')
    doc = json.loads(body)
    state = doc['State']
    require(doc['Id'] == cid and state['Status'] == 'exited' and state['Running'] is False
            and state['ExitCode'] == 1 and state['FinishedAt'] != '0001-01-01T00:00:00Z',
            'container not exact terminal failure')
    return {'id': cid, 'finished_at': state['FinishedAt'], 'exit_code': state['ExitCode']}


def put(path, doc):
    raw = (json.dumps(doc, sort_keys=True, indent=2) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == raw, 'existing reconciliation evidence differs')
    else:
        publish_new(path, raw)


def reconcile(config, socket_path='/var/run/docker.sock'):
    c = read(config)
    require(c['schema'] == 'saea-partial-score-reconciliation-v1', 'invalid schema')
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'reconciliation.guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _reconcile(c, Path(config), out, socket_path)


def _reconcile(c, config, out, socket_path):
    status = {key: terminal(c[key], socket_path) for key in ('producer_container', 'recovery_container')}
    paths = {k: Path(v['path']) for k, v in c['pins'].items()}
    for name, path in paths.items():
        if name == 'attempt':
            continue
        require(path.is_file() and not path.is_symlink() and sha(path) == c['pins'][name]['sha256'], 'pin differs: ' + name)
    for name, container in (('producer', c['producer_container']), ('recovery', c['recovery_container'])):
        launch = read(paths[name + '_launch'])
        launch_intent = read(paths[name + '_launch_intent'])
        launch_result = read(paths[name + '_launch_result'])
        digest = hashlib.sha256(json.dumps(launch, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(launch_intent['config_sha256'] == digest and launch_result['returncode'] == 0
                and launch_result['stdout'].strip() == container and launch_result['automatic_retry'] is False,
                name + ' launch receipt differs')
    intent = read(paths['recovery_intent'])
    require(intent['schema'] == 'saea-sealed-overlay-score-recovery-intent-v1' and intent['rows'] == c['rows'] and intent['new_model_calls'] == 0, 'recovery intent differs')
    for name in ('generation_manifest', 'jobs', 'ledger', 'failed_summary', 'failed_stderr', 'flight_receipts'):
        require(intent['inputs'][name] == c['pins'][name], 'recovery source binding differs: ' + name)
    require(intent['inputs']['manifest']['sha256'] == c['pins']['generation_manifest']['sha256'], 'original manifest differs')
    batch = read(paths['recovery_batch'])
    require(batch['operation'] == 'finalize' and batch['job_count'] == batch['failed'] == 1 and batch['succeeded'] == 0, 'batch is not failed finalization')
    job, = batch['jobs']
    require(job['status'] == 'failed' and job['returncode'] == 1 and job['result_file'] is None and job['complete_receipt_file'] is None and job['stderr_sha256'] == c['pins']['recovery_stderr']['sha256'], 'failed job differs')
    stderr = paths['recovery_stderr'].read_text()
    require('Read-only file system' in stderr and 'handle.mark_finalized()' in stderr and str(paths['attempt_lock']) in stderr, 'failure is not the expected attempt lock failure')
    generation = read(paths['generation_manifest'])
    manifest = read(paths['run_manifest'])
    require({k:v for k,v in generation.items() if k != 'scores'} == {k:v for k,v in manifest.items() if k != 'scores'} and generation['scores'] is None, 'generation fields changed')
    require(manifest['scores'] == {'path': str(paths['scores']), 'sha256': c['pins']['scores']['sha256']}, 'score attachment differs')
    require(manifest['attempt']['state_path'] == str(paths['attempt']), 'attempt path differs')
    require(intent['source_identities']['generation_commit'] == manifest['study_commit'], 'source identity differs')
    require(job['manifest'] == str(paths['run_manifest']) and job['ledger_cells'] == str(paths['ledger']), 'batch job identity differs')
    bundle = verify_bundle(paths['run_manifest'])
    require(bundle.attempt_state in ('OPENED', 'FINALIZED') and len(bundle.outputs_rows) == len(bundle.split_ids) == c['rows'], 'bundle state or rows differ')
    scores = read(paths['scores'])['slurp-slu-official-train']['per_sample']
    ids = [row['sample_id'] for row in scores]
    require(len(ids) == c['rows'] and len(set(ids)) == c['rows'] and set(ids) == bundle.split_ids, 'score sample identities differ')
    markdown, cells = ledger_row_for_bundle(bundle, read(paths['ledger']))
    require(read(paths['ledger_json']) == cells and paths['ledger_md'].read_text() == markdown + '\n', 'ledger differs')
    attempt = read(paths['attempt'])
    require(attempt['actual'] == c['actual'] and attempt['reserved'] == c['reserved'] and attempt['run_id'] == manifest['run_id'] and attempt['attempt_id'] == manifest['attempt']['attempt_id'], 'attempt identity or costs differ')
    if attempt['state'] == 'OPENED':
        require(sha(paths['attempt']) == c['pins']['attempt']['sha256'], 'original attempt differs')
    require(paths['attempt_lock'].is_file() and not paths['attempt_lock'].is_symlink(), 'attempt lock absent')
    evidence = {'schema':'saea-partial-score-reconciliation-intent-v1', 'config_sha256':sha(config), 'terminals':status, 'pins':c['pins'], 'actual':c['actual'], 'reserved':c['reserved'], 'new_model_calls':0}
    intent_path = out / 'intent.json'
    if bundle.attempt_state == 'FINALIZED':
        require(intent_path.is_file() and read(intent_path) == evidence, 'finalized without matching reconciliation intent')
    else:
        put(intent_path, evidence)
        handle = AttemptHandle(path=paths['attempt'], run_id=attempt['run_id'], attempt_id=attempt['attempt_id'])
        handle.mark_finalized()
    after = verify_bundle(paths['run_manifest'])
    require(after.attempt_state == 'FINALIZED', 'final bundle is not finalized')
    final_attempt = read(paths['attempt'])
    require(final_attempt['actual'] == c['actual'] and final_attempt['reserved'] == c['reserved'], 'attempt costs changed')
    result = {'schema':'saea-partial-score-reconciliation-v1', 'intent_sha256':sha(intent_path), 'manifest_sha256':sha(paths['run_manifest']), 'scores_sha256':sha(paths['scores']), 'attempt_sha256':sha(paths['attempt']), 'rows':c['rows'], 'new_model_calls':0}
    put(out / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.config), sort_keys=True))
