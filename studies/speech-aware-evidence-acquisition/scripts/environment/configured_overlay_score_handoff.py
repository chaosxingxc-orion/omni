#!/usr/bin/env python3
"""Wait for one pinned worker and hand a known scoring failure to the native launcher."""

import argparse
import copy
import fcntl
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
import configured_job_launch as launcher
SCHEMA = 'saea-overlay-score-handoff-v1'

def check(ok, msg):
    if not ok:
        raise ValueError(msg)

def pin(p):
    path = Path(p['path'])
    check(path.is_file() and launcher.digest(path) == p['sha256'], 'pinned input differs: ' + str(path))
    return path

def save(path, doc, replace=False):
    launcher.persist(path, doc, replace=replace)

def load(path):
    d = json.loads(Path(path).read_bytes())
    keys = {'schema', 'state_dir', 'producer_launch', 'producer_launcher', 'container_id', 'run_id', 'attempt_id', 'score_tool', 'tool_manifest_sha256', 'tool_script_sha256', 'official_python_sha256', 'worktree', 'worktree_commit', 'interval_seconds'}
    check(set(d) == keys and d['schema'] == SCHEMA and (d['interval_seconds'] == 600), 'invalid handoff config')
    check(Path(d['state_dir']).resolve().is_relative_to(Path('/var/lib/omni-local-state')) and len(Path(d['state_dir']).parts) >= 5, 'state directory must be private local project path')
    for name in ('run_id', 'attempt_id'):
        check(isinstance(d[name], str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,180}', d[name]), 'invalid identity')
    check(d['container_id'] is None or bool(re.fullmatch('[a-f0-9]{64}', d['container_id'])), 'invalid container')
    for name in ('state_dir', 'score_tool', 'worktree'):
        launcher.absolute(d[name])
    pin(d['producer_launch'])
    pin(d['producer_launcher'])
    check(re.fullmatch('[a-f0-9]{64}', d['tool_manifest_sha256']) and re.fullmatch('[a-f0-9]{64}', d['tool_script_sha256']) and re.fullmatch('[a-f0-9]{64}', d['official_python_sha256']) and re.fullmatch('[a-f0-9]{40}', d['worktree_commit']), 'invalid pin')
    check(launcher.digest(Path(d['score_tool']) / 'source-manifest.json') == d['tool_manifest_sha256'], 'tool manifest differs')
    check(launcher.digest(Path(d['score_tool']) / 'studies/speech-aware-evidence-acquisition/scripts/analysis/recover_overlay_score.py') == d['tool_script_sha256'], 'recovery script differs')
    return d

def map_path(producer, logical):
    return launcher.mapped_input(producer, logical)

def canonical_hash(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def producer_status(d, producer):
    path = pin(d['producer_launcher'])
    cmd = ['python3', str(path), 'status', '--config', d['producer_launch']['path']]
    r = subprocess.run(cmd, text=True, capture_output=True)
    check(r.returncode == 0, 'producer status failed: ' + r.stderr[-500:])
    s = json.loads(r.stdout)
    if s['state'] == 'not-launched':
        return s
    check(s['state'] != 'unresolved-launch-intent', 'producer launch intent unresolved')
    cid = s.get('container_id')
    check(bool(re.fullmatch('[a-f0-9]{64}', cid or '')), 'producer container missing')
    check(d['container_id'] is None or d['container_id'] == cid, 'producer container differs')
    root = Path(producer['receipt_dir'])
    intent = json.loads((root / 'launch-intent.json').read_bytes())
    result = json.loads((root / 'launch-result.json').read_bytes())
    check(intent.get('config_sha256') == canonical_hash(producer) and intent.get('runner_sha256') == d['producer_launcher']['sha256'] and (intent.get('expected_commit') == producer['expected_commit']) and (intent.get('automatic_retry') is False), 'producer intent differs')
    check(result.get('returncode') == 0 and result.get('automatic_retry') is False and (result.get('stdout', '').strip() == cid), 'producer receipt differs')
    return s

def files(d, producer):
    logical = Path('/mnt/cephfs/xingchao/omni')
    run = logical / 'runs' / d['run_id'] / d['attempt_id']
    host = map_path(producer, run)
    check(host.is_dir(), 'run directory absent')
    return (logical, run, host)

def logical_release(d):
    return Path('/mnt/cephfs/xingchao/omni/work/codex') / d['attempt_id'] / 'generation-released.json'

def sealed_inputs(d, producer, host):
    run = d['run_id']
    manifest = host / (run + '.run-manifest.json')
    gen = host / (run + '.generation-manifest.json')
    jobs = host / 'score-jobs.jsonl'
    ledger = host / 'score-ledger-cells.json'
    summary = host / 'score-batch/batch-summary.json'
    stderr = host / 'score-batch/generation-v1.stderr.log'
    receipt = host / 'flight-receipts.json'
    for p in (manifest, gen, jobs, ledger, summary, stderr, receipt):
        check(p.is_file() and (not p.is_symlink()), 'missing sealed score input: ' + p.name)
    repo = Path(d['worktree'])
    head = launcher.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'])
    clean = launcher.run(['git', '-C', str(repo), 'status', '--porcelain'])
    check(head.returncode == clean.returncode == 0 and head.stdout.strip() == d['worktree_commit'] and (not clean.stdout.strip()), 'clean scoring launcher worktree required')
    release_path = map_path(producer, logical_release(d))
    check(release_path.is_file(), 'generation release absent')
    release = json.loads(release_path.read_bytes())
    check(release.get('run_id') == run and release.get('attempt_id') == d['attempt_id'] and (release.get('study_commit') == producer['expected_commit']) and (release.get('generation_sealed') is True) and (release.get('generation_lease_released') is True) and (release.get('rows') == 1080) and (release.get('artifacts', {}).get('generation_manifest', {}).get('sha256') == launcher.digest(gen)), 'generation release differs')
    m = json.loads(manifest.read_bytes())
    check(m.get('config', {}).get('fragments', {}).get('experiment') == producer['argv'][producer['argv'].index('--candidate') + 1], 'candidate differs')
    fail = json.loads(summary.read_bytes())
    fr = json.loads(receipt.read_bytes())
    check(launcher.digest(manifest) == launcher.digest(gen), 'generation/run manifest differs')
    check(m['run_id'] == run and m['attempt']['attempt_id'] == d['attempt_id'] and (m['status'] == 'COMPLETED') and (m['scores'] is None) and (m['study_commit'] == producer['expected_commit']) and (m['split']['count'] == 1080) and (m['row_checkpoint']['complete_rows'] == 1080), 'sealed identity differs')
    check(fail.get('failed') == 1 and fail.get('succeeded') == 0 and (fail.get('job_count') == 1) and (fail.get('operation') == 'finalize') and ('delivery config differs from recorded run' in stderr.read_text()), 'not known overlay scoring failure')
    failed_job = fail.get('jobs', [])
    check(len(failed_job) == 1 and failed_job[0].get('status') == 'failed' and (failed_job[0].get('returncode') == 1) and (failed_job[0].get('result_file') is None) and (failed_job[0].get('complete_receipt_file') is None) and (failed_job[0].get('stderr_sha256') == launcher.digest(stderr)), 'failed score job identity differs')
    check(fr.get('stages', {}).get('score', {}).get('result') == fail, 'failure receipt differs')
    check(not list(host.glob('*.scores.json')) and (not list(host.glob('score-recovery*'))), 'score or recovery already exists')
    return {k: launcher.digest(v) for k, v in {'manifest': manifest, 'jobs': jobs, 'ledger': ledger, 'failed_summary': summary, 'failed_stderr': stderr, 'flight_receipts': receipt}.items()}

def finalized(d, producer, host):
    mpath = host / (d['run_id'] + '.run-manifest.json')
    check(mpath.is_file(), 'final manifest absent')
    m = json.loads(mpath.read_bytes())
    check(m.get('run_id') == d['run_id'] and m.get('study_commit') == producer['expected_commit'] and (m.get('attempt', {}).get('attempt_id') == d['attempt_id']) and (m.get('config', {}).get('fragments', {}).get('experiment') == producer['argv'][producer['argv'].index('--candidate') + 1]) and (m.get('status') == 'COMPLETED') and (m.get('split', {}).get('count') == 1080) and (m.get('row_checkpoint', {}).get('complete_rows') == 1080) and m.get('scores'), 'final score identity differs')
    scores = m['scores']
    check(isinstance(scores, dict) and isinstance(scores.get('path'), str) and isinstance(scores.get('sha256'), str), 'score attachment missing')
    sp = map_path(producer, scores['path'])
    check(sp.is_file() and launcher.digest(sp) == scores['sha256'], 'score attachment pin differs')
    return {'manifest_sha256': launcher.digest(mpath), 'scores_sha256': scores['sha256']}

def make_launch(d, producer, pins, logical, run, host):
    doc = copy.deepcopy(producer)
    for k in ('generation_lease', 'process_access', 'framework_runtime', 'fresh_outputs', 'source_identity'):
        doc.pop(k, None)
    doc.update(container_name='omni-score-handoff-' + hashlib.sha256((d['run_id'] + d['attempt_id']).encode()).hexdigest()[:20], repo_path=d['worktree'], expected_commit=d['worktree_commit'], receipt_dir=str(Path(d['state_dir']) / 'launcher'), working_directory='/tmp', network='none', pid_host=False, cpuset_cpus='72-75', user='0:1008', create_directories=[str(Path(d['state_dir']) / 'launcher')], allowed_create_roots=[d['state_dir']], monitor_interval_seconds=600)
    env = doc['environment']
    env.pop('SAEA_STAGE_TIMING_PATH', None)
    env.update(SAEA_RUNTIME_CONFIG='/config/runtime-native-c64-p8-v8.json', SAEA_SCORE_TOOL_RELEASE='/score-tool', SAEA_SCORE_TOOL_RELEASE_SHA256=d['tool_manifest_sha256'])
    mounts = []
    for m in doc['mounts']:
        target = Path(m['target'])
        m['readonly'] = True
        if target == Path('/opt/saea-release'):
            check(m['readonly'], 'generation release must be readonly')
        if target in (logical / 'runs', logical / 'attempts'):
            continue
        mounts.append(m)
    # Existing producer roots supply RO mountpoints; writable overrides are
    # limited to this run and its attempt, never the other experiments.
    run_root = map_path(producer, logical / 'runs')
    attempt_root = map_path(producer, logical / 'attempts')
    attempt_dir = attempt_root / d['run_id']
    mounts.extend([
        {'source': str(run_root), 'target': str(logical / 'runs'), 'readonly': True},
        {'source': str(attempt_root), 'target': str(logical / 'attempts'), 'readonly': True},
        {'source': str(host), 'target': str(run), 'readonly': False},
        {'source': str(attempt_dir), 'target': str(logical / 'attempts' / d['run_id']), 'readonly': False},
        {'source': d['score_tool'], 'target': '/score-tool', 'readonly': True},
    ])
    doc['mounts'] = mounts
    runstr = str(run)
    mn = runstr + '/' + d['run_id']
    doc['argv'] = ['/opt/saea-venv/bin/python', '/score-tool/studies/speech-aware-evidence-acquisition/scripts/analysis/recover_overlay_score.py', '--source-repo', '/opt/saea-release/studies/speech-aware-evidence-acquisition', '--data', str(logical), '--manifest', mn + '.run-manifest.json', '--generation-manifest', mn + '.generation-manifest.json', '--manifest-sha256', pins['manifest'], '--jobs', runstr + '/score-jobs.jsonl', '--jobs-sha256', pins['jobs'], '--ledger', runstr + '/score-ledger-cells.json', '--ledger-sha256', pins['ledger'], '--failed-summary', runstr + '/score-batch/batch-summary.json', '--failed-summary-sha256', pins['failed_summary'], '--failed-stderr', runstr + '/score-batch/generation-v1.stderr.log', '--failed-stderr-sha256', pins['failed_stderr'], '--official-python-sha256', d['official_python_sha256'], '--lock', '/opt/saea-release/docs/datasets.lock.json', '--out', runstr + '/score-recovery-runtime-overlay-v1', '--rows', '1080']
    doc['required_inputs'] = [d['producer_launch'], d['producer_launcher'], {'path': str(Path(d['score_tool']) / 'source-manifest.json'), 'sha256': d['tool_manifest_sha256']}]
    return doc

def step(d, launcher_module=launcher):
    state = Path(d['state_dir'])
    state.mkdir(parents=True, exist_ok=True)
    producer = launcher.load_config(pin(d['producer_launch']))
    check(producer['argv'].count('--candidate') == 1, 'producer candidate missing')
    check(producer['argv'].count('--out') == 1 and Path(producer['argv'][producer['argv'].index('--out') + 1]).name == d['attempt_id'], 'producer attempt differs')
    s = producer_status(d, producer)
    if s['state'] in ('not-launched', 'running', 'created', 'restarting', 'paused'):
        return {'status': 'waiting-producer', 'producer': s}
    check(s['state'] == 'exited', 'unknown producer state')
    logical, run, host = files(d, producer)
    attempt = map_path(producer, logical / 'attempts' / d['run_id'] / (d['attempt_id'] + '.json'))
    check(attempt.is_file(), 'attempt absent')
    attempt_state = json.loads(attempt.read_bytes()).get('state')
    if attempt_state == 'FINALIZED':
        if (state / 'recovery-launch.json').is_file():
            recovery = launcher_module.execute(state / 'recovery-launch.json', 'status')
            check(recovery['state'] == 'exited' and recovery.get('exit_code') == 0, 'recovery not terminal successful')
        evidence = finalized(d, producer, host)
        evidence.update(run_id=d['run_id'], attempt_id=d['attempt_id'], generation_commit=producer['expected_commit'], producer_container_id=s['container_id'])
        ready = state / 'score-ready.json'
        if ready.exists():
            check(json.loads(ready.read_bytes()) == evidence, 'score-ready identity changed')
        else:
            save(ready, evidence)
        return {'status': 'score-ready', 'source': evidence, 'producer': s}
    check(attempt_state == 'OPENED', 'unknown attempt state')
    check(s.get('exit_code') == 1, 'worker terminal is not expected failure')
    launch_path = state / 'recovery-launch.json'
    if launch_path.exists():
        observed = launcher_module.execute(launch_path, 'status')
        check(observed['state'] in ('running', 'created', 'restarting', 'paused'), 'recovery terminal or unresolved without FINALIZED; inspect retained receipt')
        return {'status': 'recovery-' + observed['state'], 'recovery': observed, 'producer': s}
    pins = sealed_inputs(d, producer, host)
    doc = make_launch(d, producer, pins, logical, run, host)
    save(launch_path, doc)
    launcher_module.execute(launch_path, 'layout')
    result = launcher_module.execute(launch_path, 'launch')
    return {'status': 'recovery-' + result['state'], 'recovery': result, 'pins': pins, 'producer': s}

def execute(path, mode):
    d = load(path)
    state = Path(d['state_dir'])
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'handoff.guard').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = step(d)
            except Exception as error:
                result = {'status': 'held', 'reason': type(error).__name__ + ': ' + str(error)}
            result.update(config_sha256=launcher.digest(path), observed_at_unix=time.time(), new_model_calls=0, automatic_retry=False)
            save(state / 'status.json', result, replace=True)
            if mode == 'once' or result['status'] in ('held', 'score-ready'):
                return result
            time.sleep(600)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['once', 'watch'])
    p.add_argument('--config', required=True)
    a = p.parse_args()
    print(json.dumps(execute(a.config, a.mode), sort_keys=True))
