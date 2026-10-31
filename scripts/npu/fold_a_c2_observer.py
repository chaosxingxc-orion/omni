"""Read full Fold A checkpoints and eight-card telemetry every 600 seconds.

No inference, job control, or automatic replay. Unknown process state is not a
failure verdict. CPU accounting reads the shared verified checkpoint reader.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D, IMAGE, status, write
from card_utilization_monitor import sample as hardware
RUN='SAEA-SLURP-A0-FOLD-A-CURRENT-C2-V1'
NAME='slurp-a0-fold-a-current-c2-v1'


def read(p):return json.loads(p.read_bytes()) if p.exists() else None


def observe(tree,run,out,terminal,container):
    for p in (tree,run,out,terminal):
        if not p.resolve().is_relative_to(D):raise ValueError('owned paths required')
    if out.exists():raise ValueError('fresh observer output required')
    out.mkdir(parents=True)
    write(out/'intent.json',{'pid':os.getpid(),'interval_seconds':600,'can_restart_jobs':False,'model_calls':0})
    study=tree/'studies/speech-aware-evidence-acquisition'
    directory=D/'runs'/RUN/'generation-v1';seal=directory/(RUN+'.generation-manifest.json')
    checkpoint=directory/(RUN+'.rows.ckpt.jsonl');roster=study/'configs/dataset/slurp-speech-native-evo-fold-a-1080.json'
    while True:
        now=time.time()
        try:
            start=read(run/'started.json');exit_code=terminal.read_text().strip() if terminal.exists() else None
            process=subprocess.run(['docker','inspect','--format','{{.State.Running}}',container],capture_output=True,text=True)
            live=process.returncode==0 and process.stdout.strip()=='true'
            sealed=read(seal)
            complete=bool(sealed and sealed.get('status')=='COMPLETED' and sealed.get('run_id')==RUN)
            state='complete' if complete else ('failed' if exit_code is not None and exit_code!='0' else ('running' if live else 'unconfirmed_process_state'))
            result={'completed_rows':0 if not checkpoint.exists() else None,'total_rows':1080,'remaining_rows':1080 if not checkpoint.exists() else None,
                'cards':8,'instances':2,'wall_seconds_per_complete_row':None,'estimated_remaining_seconds':None}
            if start:
                if start['candidate']!=NAME or start['rows']!=1080:raise ValueError('unexpected run boundary')
                if checkpoint.exists() and state!='unconfirmed_process_state':
                    target=out/('snapshot-'+str(time.time_ns())+'.json')
                    env={**os.environ,'TREE':str(tree),'CPUSET':'176-191','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,
                        'PORT_DOCKER_FLAGS':'--network none --name saea-fold-a-c2-observation-v1'}
                    command=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/real_engine_progress.py',
                        '--checkpoint',str(checkpoint),'--roster',str(roster),'--state',state,
                        '--elapsed-seconds',str(max((seal.stat().st_mtime if complete else now)-start['time'],.001)),
                        '--cards','8','--instances','2','--output',str(target)]
                    with (out/'reader.log').open('a') as log:subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
                    result=read(target);result.pop('state');eta=result.pop('estimated_remaining_seconds')
                    result['linear_remaining_seconds']=eta
                    result['estimated_remaining_seconds']=eta if complete or 100<=result['completed_rows']<=972 else None
            telemetry=hardware(out)
            with (out/'hardware.jsonl').open('a') as f:f.write(json.dumps(telemetry)+'\n')
            status(out,state,**result,observed_at=now,generation_container_live=live,generation_sealed=complete,
                campaign_exit=exit_code,hardware=telemetry,resource_note='8 allocated cards, 2 TP4 instances; utilization is observed separately.')
            if exit_code is not None:return
        except Exception as error:
            status(out,'observation_error',observed_at=now,error=str(error),job_restarted=False)
        time.sleep(600)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('tree','run','out','terminal'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--container',required=True)
    observe(**vars(p.parse_args()))
