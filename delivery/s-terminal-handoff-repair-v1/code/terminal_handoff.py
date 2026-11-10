"""Narrow, fail-closed FAILED/PAUSED predecessor handoff. No request delivery."""
import hashlib
import json
import os
from pathlib import Path
import time

SCHEMA = 's-terminal-handoff-v1'
STATUS = 'RECONCILED_FOR_INDEPENDENT_SUCCESSOR_ONLY'


def need(ok, reason):
    if not ok:
        raise RuntimeError('BLOCKED_TERMINAL_HANDOFF: ' + reason)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path):
    return dict(path=str(path), sha256=digest(Path(path).read_bytes()))


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    syncdir(Path(path).parent)


def json_new(path, doc):
    write_new(path, json.dumps(doc, sort_keys=True).encode() + b'\n')


def dead(pid):
    need(type(pid) is int and pid > 0, 'missing owner pid')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False  # PermissionError is an explicit refusal, never proof of death.


def check_policy(policy, task, evidence):
    need(policy.get('schema') == SCHEMA and policy.get('enabled') is True, 'disabled')
    need(evidence['terminal']['status'] in ('FAILED', 'PAUSED'), 'not a known failed terminal')
    need(task['task_id'] != evidence['launch']['task_id'], 'same task replay')
    need(evidence['launch']['task_id'] not in task.get('dependencies', []), 'dependent successor')
    # Exact successor -> exact failed task IDs. No wildcard/default/lane expansion.
    allowed = policy.get('allowlist', {}).get(task['task_id'], [])
    need(isinstance(allowed, list) and evidence['launch']['task_id'] in allowed, 'pair not allowlisted')


def reconcile(root, task, cfg, rt, evidence, policy, held, verify, metrics, inspect):
    """Caller holds the original shared guard until capture is durable.

    Probe functions are host_driver's existing local identity/engine probes.
    Archive must be empty; any interrupted transaction requires owner review.
    """
    check_policy(policy, task, evidence)
    root = Path(root)
    launch = evidence['launch']
    failed = root / 'jobs' / launch['task_id']
    need(failed.resolve() == Path(launch['state']).resolve(), 'failed state outside exact jobs path')
    sources = {name: failed / rel for name, rel in {
        'launch.json': 'LAUNCH.json', 'terminal.json': 'run/TERMINAL.json',
        'runtime-config.json': 'runtime-config.json'}.items()}
    raw = {name: path.read_bytes() for name, path in sources.items()}
    need(json.loads(raw['launch.json']) == launch, 'launch evidence changed')
    term = json.loads(raw['terminal.json'])
    need(term == evidence['terminal'], 'terminal evidence changed')
    oldcfg = json.loads(raw['runtime-config.json'])
    for key in ('run_id', 'attempt_id'):
        need(oldcfg[key] == launch[key], 'failed config/launch ' + key)
    need(cfg['run_id'] != oldcfg['run_id'], 'same run replay')
    for key in ('boot_id', 'session_id', 'service_id', 'service_pid', 'shared_guard', 'shared_guard_inode'):
        need(rt[key] == launch['runtime'][key] == oldcfg['runtime'][key], 'failed runtime changed: ' + key)
    guard = Path(rt['shared_guard'])
    flight = guard.parent / 'flight.lock'
    need(not flight.is_symlink(), 'symlink flight')
    flight_raw = flight.read_bytes() if flight.exists() else None
    flight_stat = flight.stat() if flight_raw is not None else None
    doc = json.loads(flight_raw) if flight_raw is not None else None
    if doc is not None:
        need(doc.get('schema') == 'saea-generation-lease-v1', 'unsupported lease')
        for key, expected in dict(run_id=oldcfg['run_id'], attempt_id=oldcfg['attempt_id'],
                                  session_id=rt['session_id'], study_head=rt['study_head'],
                                  owner='s-correct-p-data-v1', run_ids=[oldcfg['run_id']]).items():
            need(doc.get(key) == expected, 'flight not failed run: ' + key)
        need(isinstance(doc.get('token'), str) and len(doc['token']) == 64, 'invalid lease token')
        need(doc.get('pid') == launch['pid'], 'flight owner not failed runner')

    def check():
        verify(rt, task['cards'])
        need(os.fstat(held.fileno()).st_ino == guard.stat().st_ino == rt['shared_guard_inode'], 'guard inode changed')
        need(os.fstat(held.fileno()).st_dev == guard.stat().st_dev, 'guard device changed')
        # Confirm lock is actually held, not just an open descriptor.
        import fcntl
        with guard.open('r+b') as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                need(False, 'guard not held')
        old = inspect(launch['container_id'])
        need(old['Id'] == launch['container_id'] and old['State'].get('Status') == 'exited'
             and old['State']['Running'] is False and old['State']['Pid'] == 0, 'old container not exited')
        need(dead(launch['pid']) and (doc is None or dead(doc['pid'])), 'live runner/flight owner')
        need(not (guard.parent / 'engine-transition.pending.json').exists(), 'engine transition')
        need(metrics(rt) == dict(running=0, waiting=0), 'engine busy')
        for name, path in sources.items():
            need(path.read_bytes() == raw[name], 'source changed: ' + name)
        if doc is None:
            need(not flight.exists() and not flight.is_symlink(), 'flight appeared')
        else:
            now = flight.stat()
            need((now.st_dev, now.st_ino) == (flight_stat.st_dev, flight_stat.st_ino)
                 and flight.read_bytes() == flight_raw, 'flight changed')

    check()
    parent = root / 'reconciliation'
    parent.mkdir(exist_ok=True)
    syncdir(root)
    archive = parent / digest((oldcfg['run_id'] + '\0' + oldcfg['attempt_id'] + '\0' + task['task_id']).encode())
    need(not archive.exists(), 'partial/existing archive; explicit owner review required')
    archive.mkdir()
    syncdir(parent)
    # Copy, fsync, read-back SHA, then unlink only metadata. Never rename across mounts.
    if doc is not None:
        raw['flight.lock'] = flight_raw
    else:
        raw['NO-FLIGHT.json'] = json.dumps(dict(schema=SCHEMA, flight_present=False,
            failed_run_id=oldcfg['run_id'], failed_attempt_id=oldcfg['attempt_id'],
            runner_pid=launch['pid'], guard_inode=rt['shared_guard_inode'],
            reason='no metadata under held original guard; exited runner and idle engine verified')).encode()+b'\n'
    for name, data in raw.items():
        write_new(archive / name, data)
        need(digest((archive / name).read_bytes()) == digest(data), 'partial archive/hash mismatch: ' + name)
    receipt = dict(schema=SCHEMA, status=STATUS, at=time.time(), successor_task_id=task['task_id'],
                   successor_run_id=cfg['run_id'], successor_attempt_id=cfg['attempt_id'],
                   failed_task_id=launch['task_id'], failed_run_id=oldcfg['run_id'], failed_attempt_id=oldcfg['attempt_id'],
                   terminal=term, unknown_preserved=term['unknown'], unknown_replayed=False,
                   runtime={k:rt[k] for k in ('boot_id','session_id','service_id','service_pid','shared_guard','shared_guard_inode')},
                   archive={name:pin(archive / name) for name in raw},
                   failed_container_id=launch['container_id'], runner_pid=launch['pid'], flight_present=doc is not None,
                   flight_owner_pid=doc['pid'] if doc is not None else None,
                   engine_metrics=dict(running=0, waiting=0), old_container_exited=True)
    json_new(archive / 'PREPARED.json', receipt)
    check()
    for name, data in raw.items():
        need(digest((archive / name).read_bytes()) == digest(data), 'archive changed before unlink: ' + name)
    if doc is not None:
        flight.unlink()
        syncdir(guard.parent)
    json_new(archive / 'RECEIPT.json', receipt)
    return pin(archive / 'RECEIPT.json')


def validate_predecessor(pred, runtime, config):
    """Called by both native runners before model admission; COMPLETE stays unchanged."""
    if pred.get('status') == 'COMPLETE' and pred.get('unknown') == 0:
        return
    need(pred.get('status') in ('FAILED','PAUSED'), 'unsupported predecessor terminal')
    p = runtime['terminal_handoff_receipt']
    need(digest(Path(p['path']).read_bytes()) == p['sha256'], 'handoff receipt pin')
    receipt = read(p['path'])
    need(receipt['schema'] == SCHEMA and receipt['status'] == STATUS and receipt['unknown_replayed'] is False,
         'handoff receipt status')
    need(receipt['terminal'] == {k:v for k,v in pred.items() if k not in ('container','container_state','terminal')}, 'terminal custody')
    need(receipt['successor_run_id'] == config['run_id'] and receipt['successor_attempt_id'] == config['attempt_id']
         and receipt['failed_run_id'] != config['run_id'], 'independent successor identity')
    need(receipt['failed_container_id'] == pred['container'], 'predecessor container')
    for key, value in receipt['runtime'].items():
        need(runtime[key] == value, 'receipt runtime changed: ' + key)
    for item in receipt['archive'].values():
        need(pin(item['path']) == item, 'archive pin')
    need(read(receipt['archive']['terminal.json']['path']) == receipt['terminal'], 'archived terminal')
    if receipt['flight_present']:
        flight = read(receipt['archive']['flight.lock']['path'])
        need((flight['run_id'], flight['attempt_id'], flight['session_id']) ==
             (receipt['failed_run_id'], receipt['failed_attempt_id'], runtime['session_id']), 'archived flight identity')
    else:
        noflight = read(receipt['archive']['NO-FLIGHT.json']['path'])
        need(noflight['flight_present'] is False and receipt['flight_owner_pid'] is None and
             (noflight['failed_run_id'],noflight['failed_attempt_id'],noflight['runner_pid'],noflight['guard_inode']) ==
             (receipt['failed_run_id'],receipt['failed_attempt_id'],receipt['runner_pid'],runtime['shared_guard_inode']), 'no-flight evidence')
