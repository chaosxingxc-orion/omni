"""Export each finalized real candidate once, independently of the NPU queue.

Watch only terminal files at600s; preserve failed exports and never retry model
or scoring jobs. Runs in a detached host process, exports in the fixed container.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

D=Path('/mnt/cephfs/xingchao/omni')
IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'


def status(root,state,**extra):
    path=root/'status.tmp'
    path.write_text(json.dumps({'state':state,'pid':os.getpid(),'time':time.time(),**extra})+'\n')
    path.replace(root/'status.json')


def run(root):
    if root.exists() or not root.resolve().is_relative_to((D/'work/codex').resolve()):
        raise ValueError('fresh owned output required')
    root.mkdir()
    items=[('slurp-real500-tp4-s128-v1','real500-successors'),('slurp-real500-tp8-s256-v1','real500-successors-tp8')]
    env={**os.environ,'TREE':str(D/'publish/omni'),'CPUSET':'32-47','PORT_CONTAINER_USER':'0:0',
         'PORT_IMAGE':IMAGE,'PORT_DOCKER_FLAGS':'--network none --name saea-real500-result-export'}
    for candidate,tree in items:
        item=D/'work/codex/real500-successor-queue-v1'/candidate
        terminal=item/'run.exit'
        status(root,'waiting_terminal',candidate=candidate)
        while not terminal.exists():
            time.sleep(600)
        if terminal.read_text().strip()!='0':
            status(root,'failed_predecessor',candidate=candidate,automatic_retry=False)
            return
        output=root/(candidate+'.json')
        args=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/export_real_engine_result.py',
              '--repo',str(D/'work/dev'/tree/'studies/speech-aware-evidence-acquisition'),'--data',str(D),
              '--candidate',candidate,'--started-path',str(item/'run/started.json'),'--out',str(output)]
        status(root,'exporting',candidate=candidate)
        with (root/(candidate+'.log')).open('x') as log:
            result=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status(root,'export_failed',candidate=candidate,exit_code=result.returncode,automatic_retry=False)
            return
        report=json.loads(output.read_bytes())
        if report['rows']!=500 or not report['official_finalized']:
            raise ValueError('incomplete export')
    status(root,'complete',model_calls=0,scoring_jobs=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    run(args.out)
