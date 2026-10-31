"""Report real K4 sample progress every 600 seconds, with no restarts or requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D, U, IMAGE, write, status


def read(path):
    return json.loads(path.read_bytes()) if path.exists() else None


def observe(tree,queue,out):
    if out.exists() or not out.resolve().is_relative_to(D/'work/codex'):
        raise ValueError('fresh owned observer required')
    study=tree/'studies/speech-aware-evidence-acquisition'
    out.mkdir(parents=True)
    write(out/'intent.json',{'pid':os.getpid(),'interval_seconds':600,'new_model_calls':0,'job_restarts':False})
    for half,n in [('h1',313),('h2',312)]:
        recipe=read(study/f'docs/readiness/slurp-tfrl-current-seed-i1-{half}-v1/analysis-recipe-v1.json')
        roster=read(study/recipe['roster']['relpath'])
        write(out/(half+'-roster.json'),{'count':n,'items':roster['items'],'split_identity_hash':recipe['split_identity_hash']})
    while True:
        now=time.time()
        try:
            results={}
            for half,n in [('h1',313),('h2',312)]:
                start=read(queue/(half+'-run/started.json'))
                if start is None:
                    results[half]={'state':'not_started','completed_rows':0,'total_rows':n,'remaining_rows':n,
                        'cards':0,'instances':0,'estimated_remaining_seconds':None};continue
                run=f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{half.upper()}-V1';directory=D/'runs'/run/'generation-v1'
                seal=directory/(run+'.generation-manifest.json');checkpoint=directory/(run+'.rows.ckpt.jsonl')
                sealed=seal.exists() and read(seal).get('status')=='COMPLETED'
                terminal=read(queue/(half+'-generation_exit.json'))
                observation=subprocess.run(['docker','inspect','--format','{{.State.Running}}',
                    f'saea-tfrl-current-{half}-generation-v1'],capture_output=True,text=True)
                live=observation.returncode==0 and observation.stdout.strip()=='true'
                state='complete' if sealed else 'failed' if terminal and terminal['exit'] else 'running' if live else 'unconfirmed'
                if state=='unconfirmed' or not checkpoint.exists():
                    results[half]={'state':state,'completed_rows':None if checkpoint.exists() else 0,'total_rows':n,
                        'cards':8 if live else None,'instances':2 if live else None,'estimated_remaining_seconds':None};continue
                target=out/(half+'-'+str(time.time_ns())+'.json')
                env={**os.environ,'TREE':str(U),'CPUSET':'224-239','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,
                    'PORT_DOCKER_FLAGS':'--network none --name saea-tfrl-current-progress-reader-v1'}
                command=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/real_engine_progress.py',
                    '--checkpoint',str(checkpoint),'--roster',str(out/(half+'-roster.json')),'--state',state,
                    '--elapsed-seconds',str(max((seal.stat().st_mtime if sealed else now)-start['time'],.001)),
                    '--cards','8','--instances','2','--output',str(target)]
                with (out/'reader.log').open('a') as log:
                    subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
                result=read(target);eta=result.pop('estimated_remaining_seconds')
                result['linear_remaining_seconds']=eta
                result['estimated_remaining_seconds']=eta if sealed or 100<=result['completed_rows']<=int(.9*n) else None
                result['generation_container_live']=live;results[half]=result
            q=read(queue/'status.json') or {}
            counts=[v['completed_rows'] for v in results.values()]
            status(out,q.get('state','waiting_queue'),observed_at=now,halves=results,total_rows=625,
                completed_rows=sum(counts) if all(x is not None for x in counts) else None,
                resource_note='Both halves share 8 cards and 2 TP4 instances sequentially; do not sum their allocated cards.',
                estimated_remaining_seconds=None)
            if q.get('state') in ('complete','failed'):return
        except Exception as error:
            status(out,'observation_error',error=str(error),observed_at=now,job_restarted=False)
        time.sleep(600)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('tree','queue','out'):p.add_argument('--'+k,type=Path,required=True)
    observe(**vars(p.parse_args()))
