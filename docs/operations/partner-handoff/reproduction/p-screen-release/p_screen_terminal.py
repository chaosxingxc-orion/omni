"""Observe exactly one screen process and record the terminal outcome, without owning its requests."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from p_screen_status import report, write_status


def info(unit):
    fields = 'Id,InvocationID,ExecMainPID,ExecMainExitTimestamp,ExecMainCode,ExecMainStatus,ActiveState,Result'
    proc = subprocess.run(['systemctl', '--user', 'show', unit, '-p', fields],
                          check=True, text=True, capture_output=True)
    return dict(line.split('=', 1) for line in proc.stdout.splitlines())


def observe(config, unit, invocation, pid, *, receipt=None):
    initial = info(unit)
    if initial['InvocationID'] != invocation or int(initial['ExecMainPID']) != pid:
        raise ValueError('screen invocation changed')
    if not initial['ExecMainExitTimestamp']:
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            pass
        else:
            try:
                if info(unit)['InvocationID'] != invocation:
                    raise ValueError('screen process changed before observer attached')
                poll = select.poll()
                poll.register(fd, select.POLLIN)
                poll.poll()
            finally:
                os.close(fd)
    # pidfd becomes readable before systemd necessarily records the unit exit.
    # The receipt triggers resource handoff, so wait for the manager's terminal
    # state as well as the process exit.
    deadline = time.monotonic()+30
    while True:
        last = info(unit)
        if last['ActiveState'] in ('inactive','failed'):
            break
        if last['InvocationID'] != invocation or time.monotonic() >= deadline:
            raise ValueError('screen unit did not settle after its process exited')
        time.sleep(.25)
    observed = datetime.datetime.now().astimezone()
    if last['InvocationID'] and (last['InvocationID']!=invocation or
            (int(last['ExecMainPID']) not in (0,pid))):
        raise ValueError('screen invocation changed at process exit')
    exit_status = last['ExecMainStatus'] if last['ExecMainExitTimestamp'] else None
    data = report(config)
    root = Path(config['output'])
    seal = root / 'COMPLETE.json'
    if data['actual_complete']:
        if last['Result'] != 'success' or exit_status not in (None,'0'):
            raise ValueError('screen wrote a seal but the process failed')
        exhaust = config['sampling'].get('completion') == 'exhaust_queue'
        roster = root / ('P-DELTA-SCORED.private.jsonl' if exhaust else 'P-SAMPLE.private.jsonl')
        digest_key = 'scored_sha256' if exhaust else 'roster_sha256'
        if hashlib.sha256(roster.read_bytes()).hexdigest() != json.loads(seal.read_bytes())[digest_key]:
            raise ValueError('sealed roster digest differs')
        end = data['completed_at']
        outcome = 'delta_sealed' if exhaust else 'roster_sealed'
    else:
        end = observed.isoformat(timespec='seconds')
        # Exhaustion is only demonstrated when no remaining candidate or
        # unresolved call could fill an outstanding stratum.
        deficits = [v for v in data.get('population_yield', {}).values() if any(v['remaining_quota'].values())]
        exhausted = any(v['remaining_candidates_including_inflight'] == 0 for v in deficits)
        unresolved = sum(data['status'].get(s, 0) for s in ('intent', 'unknown'))
        outcome = 'candidate_pool_exhausted' if exhausted and not unresolved else 'stopped_before_seal'
    data.update(outcome=outcome, terminal_at=end, terminal_time_source='seal' if data['actual_complete'] else 'observed_process_exit',
                observed_at=observed.isoformat(), screen_invocation=invocation, screen_pid=pid,
                screen_unit=info(unit)['Id'], exit_status=exit_status,
                screen_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                source_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    write_status(receipt or root/'TERMINAL-TIME.json', data)
    return data


def continue_screen(config, unit, invocation, pid, successor_config, successor_unit):
    successor = json.loads(successor_config.read_bytes())
    target = Path(successor['output'])
    if (config['output'] == successor['output'] or config['hosts'] != successor['hosts']
            or config['model_bindings_by_port'] != successor['model_bindings_by_port']
            or config['recall_delta']['bank'] != successor['recall_delta']['bank']):
        raise ValueError('successor must use a separate ledger and the same qualified services and bank')
    result = observe(config, unit, invocation, pid, receipt=target/'PREDECESSOR-TERMINAL.json')
    if result['outcome'] != 'delta_sealed':
        raise ValueError('predecessor did not seal; reconcile before starting the next screen')
    if (target/'p-screen.sqlite').exists() or (target/'COMPLETE.json').exists():
        raise ValueError('successor already submitted; inspect its original unit')
    work = Path(__file__).resolve().parent
    subprocess.run(['systemd-run','--user','--service-type=exec','--unit='+successor_unit,'--working-directory='+str(work),
                    sys.executable,'-Bu',str(work/'p_screen_run.py'),str(successor_config)],check=True)
    state = info(successor_unit)
    subprocess.run(['systemd-run','--user','--unit='+successor_unit.removesuffix('.service')+'-terminal',
                    sys.executable,'-Bu',str(Path(__file__).resolve()),str(successor_config),
                    '--unit',state['Id'],'--invocation',state['InvocationID'],
                    '--pid',state['ExecMainPID']],check=True)


def finalize(plan):
    from p_screen_export import project_v2
    paths=[]
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for stage in plan['stages']:
        path=Path(stage['config']);config=json.loads(path.read_bytes())
        terminal=json.loads(Path(stage['terminal']).read_bytes())
        unit=stage['unit'];state=info(unit)
        if not (terminal['outcome']=='delta_sealed' and terminal['actual_complete']
                and terminal['source_boot']==boot and terminal['screen_unit']==unit
                and terminal['screen_config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
                and (not state['InvocationID'] or state['InvocationID']==terminal['screen_invocation'])
                and state['ActiveState']=='inactive' and state['Result']=='success'
                and config['sampling']['completion']=='exhaust_queue'):
            raise ValueError('full screen did not exit with a sealed terminal: '+unit)
        paths.append(path)
    return project_v2(paths,Path(plan['output']),selection_policy=plan['selection_policy'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--unit')
    p.add_argument('--invocation')
    p.add_argument('--pid', type=int)
    p.add_argument('--finalize', action='store_true')
    p.add_argument('--successor-config', type=Path)
    p.add_argument('--successor-unit')
    a = p.parse_args()
    config = json.loads(a.config.read_bytes())
    if a.finalize:
        print(json.dumps(finalize(config)))
    else:
        if not (a.unit and a.invocation and a.pid):
            p.error('provide the original unit, invocation and PID')
        if bool(a.successor_config) != bool(a.successor_unit):
            p.error('provide both successor parameters')
        if a.successor_config:
            continue_screen(config, a.unit, a.invocation, a.pid, a.successor_config, a.successor_unit)
        else:
            observe(config, a.unit, a.invocation, a.pid)
