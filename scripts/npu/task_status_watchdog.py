"""Detached ten-minute task status observer; no model calls or job restarts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def atomic(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def observe(config):
    result={}
    for task in config['tasks']:
        terminal=Path(task['exit'])
        if terminal.exists():
            code=terminal.read_text().strip()
            state='succeeded' if code=='0' else ('owner_stopped' if task.get('owner_stop') and Path(task['owner_stop']).exists() else 'failed')
            result[task['id']]={'state':state,'exit':code,'terminal_mtime':terminal.stat().st_mtime}
        else: result[task['id']]={'state':'pending_or_running','exit':None}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text())
    interval=config.get('interval_seconds',600)
    if interval<600:raise ValueError('owner requires at least ten-minute routine polling')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(a.out/'pid.json',{'pid':os.getpid(),'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest()})
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        previous=None
        if (a.out/'status.json').exists():previous=json.loads((a.out/'status.json').read_text()).get('tasks')
        while True:
            current=observe(config);stamp=time.time()
            atomic(a.out/'status.json',{'time':stamp,'interval_seconds':interval,'tasks':current,'model_calls':0})
            if current!=previous:
                with (a.out/'events.jsonl').open('a') as f:f.write(json.dumps({'time':stamp,'tasks':current})+'\n')
                previous=current
            time.sleep(interval)


if __name__=='__main__':main()
