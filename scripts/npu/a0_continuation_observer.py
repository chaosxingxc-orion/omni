"""Observe exact A0 continuation every600s without starting or retrying jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D, U, IMAGE, write, status

RUN = 'SAEA-SLURP-A0-REMAINING353-C2-V1'
ATTEMPT = 'attempt-a0-remaining353-c2-v1'


def observe(out, handoff=None):
    if out.exists() or not out.resolve().is_relative_to(D / 'work/codex'):
        raise ValueError('fresh owned observer required')
    out.mkdir(parents=True)
    write(out / 'intent.json', {'pid': os.getpid(), 'interval_seconds': 600,
        'new_model_calls': 0, 'can_restart_jobs': False,
        'handoff':str(handoff or D/'work/codex/a0-after-real500-queue-v1/handoff')})
    recovered = handoff is not None
    item = handoff or D / 'work/codex/a0-after-real500-queue-v1/handoff'
    if not item.resolve().is_relative_to(D/'work/codex'): raise ValueError('owned handoff required')
    directory = D / 'runs' / RUN / ATTEMPT
    checkpoint = directory / (RUN + '.rows.ckpt.jsonl')
    seal = directory / (RUN + '.generation-manifest.json')
    roster = D / 'work/dev/a0-remaining353-runtime/studies/speech-aware-evidence-acquisition/configs/dataset/slurp-a0-remaining353-c2-v1.json'
    read = lambda p: json.loads(p.read_bytes()) if p.exists() else None
    while True:
        now = time.time()
        try:
            handoff = read(item / 'status.json') or {}
            parent = {} if recovered else read(item.parent / 'status.json') or {}
            start = read(item / 'run/started.json')
            terminal = read(item / 'run.exit.json')
            failed = handoff.get('state') == 'failed' or parent.get('state') == 'failed'
            if start is None:
                status(out, 'failed_before_dispatch' if failed else 'waiting_dispatch',
                    completed_rows=0, total_rows=353, remaining_rows=353,
                    cards=0, instances=0, handoff_state=handoff.get('state'), observed_at=now,
                    estimated_remaining_seconds=None)
                if failed: return
            else:
                if start['candidate'] != 'slurp-a0-remaining353-c2-v1' or start['rows'] != 353:
                    raise ValueError('unexpected continuation identity')
                sealed = seal.exists() and read(seal).get('status') == 'COMPLETED'
                check = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}',
                    'saea-a0-remaining353-generation'], capture_output=True, text=True)
                live = check.returncode == 0 and check.stdout.strip() == 'true'
                state = 'complete' if sealed else ('failed' if failed else ('running' if live else 'unconfirmed_process_state'))
                end = seal.stat().st_mtime if sealed else now
                if checkpoint.exists() and state != 'unconfirmed_process_state':
                    target = out / ('snapshot-' + str(time.time_ns()) + '.json')
                    env = {**os.environ, 'TREE': str(U), 'CPUSET': '176-191', 'PORT_CONTAINER_USER': '0:0',
                        'PORT_IMAGE': IMAGE, 'PORT_DOCKER_FLAGS': '--network none --name saea-a0-continuation-observation-v1'}
                    command = ['bash', str(D / 'work/scripts/port_py.sh'), 'scripts/analysis/real_engine_progress.py',
                        '--checkpoint', str(checkpoint), '--roster', str(roster), '--state', state,
                        '--elapsed-seconds', str(max(end - start['time'], .001)), '--cards', '8', '--instances', '2', '--output', str(target)]
                    with (out / 'reader.log').open('a') as log:
                        subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
                    result = read(target)
                    eta = result.pop('estimated_remaining_seconds')
                    result.pop('state'); result.pop('pid', None)
                    n = result['completed_rows']
                    reliable_window = 100 <= n <= 318
                    status(out, state, **result, observed_at=now, generation_sealed=sealed,
                        generation_container_live=live, linear_remaining_seconds=eta,
                        estimated_remaining_seconds=eta if sealed or reliable_window else None,
                        eta_caution='Linear rate only; suppressed during startup and final tail.',
                        official_scoring_state='succeeded' if terminal and terminal.get('exit') == 0 else 'pending_or_failed')
                else:
                    status(out, state, completed_rows=0 if not checkpoint.exists() else None,
                        total_rows=353, observed_at=now, generation_container_live=live,
                        cards=8 if live else None, instances=2 if live else None, estimated_remaining_seconds=None)
                if terminal is not None or failed: return
        except Exception as error:
            status(out, 'observation_error', error=str(error), observed_at=now, job_restarted=False)
        time.sleep(600)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--handoff',type=Path)
    observe(**vars(p.parse_args()))
