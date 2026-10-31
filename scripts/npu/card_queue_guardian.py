#!/usr/bin/env python3
"""Supervise a detached queue runner; restart the supervisor, never retry failed jobs.

Launch with nohup setsid. SSH loss has no effect. A dead runner is recovered with
--resume, which adopts surviving jobs. Terminal stop/failure and invalid state
require reconciliation. This is not a node-reboot service or a worker retry engine.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import card_queue as cq


def recovery_action(state_path):
    path = Path(state_path)
    if not path.exists():
        return 'start'
    state = cq.read_json(str(path))
    if not isinstance(state, dict) or state.get('schema') != cq.STATE_SCHEMA:
        raise cq.QueueError('invalid existing state; preserve and reconcile')
    if state.get('state') in ('done', 'failed', 'stopped'):
        return 'terminal'
    if state.get('state') not in ('running', 'interrupted'):
        raise cq.QueueError('unknown queue state; preserve and reconcile')
    return 'resume'


def supervise(queue, directory, *, poll_s=15, restart_delay=5, max_restarts=20):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stop = [False]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    with (directory / 'guardian.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 5
        restarts = 0
        def record(state, **fields):
            cq.write_atomic(str(directory / 'guardian.json'), json.dumps(dict(
                state=state, pid=os.getpid(), updated=cq.iso(), restarts=restarts, **fields))+'\n')
        while not stop[0]:
            try:
                action = recovery_action(directory / 'state.json')
            except cq.QueueError as error:
                record('needs-reconciliation', reason=str(error))
                return 5
            if action == 'terminal':
                record('terminal', queue_state=cq.read_json(str(directory / 'state.json'))['state'])
                return 0
            # A surviving runner holds this lock, including after guardian loss.
            with (directory / 'runner.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    record('observing-existing-runner')
                    time.sleep(poll_s)
                    continue
            cmd = [sys.executable, str(Path(cq.__file__).resolve()), 'run', '--queue', str(Path(queue).resolve()),
                   '--queue-dir', str(directory), '--poll-s', str(poll_s)]
            if action == 'resume':
                cmd.append('--resume')
            with (directory / 'queue.log').open('ab') as log:
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            record('supervising', runner_pid=child.pid, action=action)
            while child.poll() is None and not stop[0]:
                time.sleep(poll_s)
            if stop[0]:
                record('interrupted', runner_pid=child.pid, reason='runner and jobs left running')
                return 3
            code = child.returncode
            if code in (0, 2, 4, 5):
                record('terminal' if code != 5 else 'needs-reconciliation', runner_exit=code)
                return code
            restarts += 1
            if restarts > max_restarts:
                record('restart-limit', runner_exit=code)
                return 6
            record('recovering-runner', runner_exit=code)
            time.sleep(restart_delay)
        record('interrupted', reason='runner and jobs left running')
        return 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', required=True)
    p.add_argument('--queue-dir', required=True)
    p.add_argument('--poll-s', type=float, default=15)
    p.add_argument('--restart-delay', type=float, default=5)
    p.add_argument('--max-restarts', type=int, default=20)
    a = p.parse_args()
    if a.poll_s <= 0 or a.restart_delay < 0 or a.max_restarts < 0:
        p.error('invalid poll, restart delay or restart limit')
    return supervise(a.queue, a.queue_dir, poll_s=a.poll_s,
                     restart_delay=a.restart_delay, max_restarts=a.max_restarts)


if __name__ == '__main__':
    sys.exit(main())
