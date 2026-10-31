"""Wait for both complete real K4 reads, then run the registered CPU pool once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D,IMAGE,write,status


def ready(completions,queue_state):
    if all(c is not None for c in completions):
        for half,n,c in zip(('H1','H2'),(313,312),completions):
            if (c.get('run_id')!=f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{half}-V1'
                    or c.get('rows')!=n or c.get('official_scoring_jobs')!=7 or c.get('new_model_calls')!=0):
                raise ValueError('unexpected half completion')
        return True
    if queue_state.get('state')=='failed':raise ValueError('upstream queue failed; preserve partial results')
    return False


def run(tree,source,queue,out):
    if out.exists() or not out.resolve().is_relative_to(D/'work/codex'):raise ValueError('fresh owned pool queue required')
    study=tree/'studies/speech-aware-evidence-acquisition';src=source/'studies/speech-aware-evidence-acquisition'
    recipe=study/'docs/readiness/tfrl-current-seed-i1-pool-read/analysis-recipe-v1.json'
    digest=hashlib.sha256(recipe.read_bytes()).hexdigest()
    commit=subprocess.check_output(['git','-C',str(study),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(study),'status','--porcelain']).strip():raise ValueError('clean analysis source required')
    target=D/'analysis/SAEA-SLURP-TFRL-CURRENT-SEED-I1-POOL-READ-V1'
    if target.exists():raise ValueError('pool already attempted')
    out.mkdir(parents=True);write(out/'intent.json',{'pid':os.getpid(),'source_commit':commit,
        'recipe_sha256':digest,'new_model_calls':0,'new_scoring_jobs':0,'interval_seconds':600})
    read=lambda p:json.loads(p.read_bytes()) if p.exists() else None
    status(out,'waiting_two_complete_halves')
    try:
        while not ready([read(D/'analysis'/f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{h}-V1-READ/complete.json') for h in ('H1','H2')],read(queue/'status.json') or {}):
            time.sleep(600)
        if hashlib.sha256(recipe.read_bytes()).hexdigest()!=digest:raise ValueError('registered pool recipe changed')
        status(out,'verifying_and_pooling')
        env={**os.environ,'TREE':str(tree),'CPUSET':'224-239','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,
            'PORT_DOCKER_FLAGS':'--network none --name saea-tfrl-current-pool-read-v1'}
        command=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/tfrl_current_pool_read.py','run',
            '--repo',str(study),'--source',str(src),'--data',str(D)]
        with (out/'read.log').open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
        write(out/'exit.json',{'exit':result.returncode,'time':time.time()})
        if result.returncode:raise ValueError('pool read failed; no automatic replay')
        complete=read(target/'complete.json')
        if complete['rows']!=625 or complete['claims_benefit'] is not False:raise ValueError('unexpected pool conclusion')
        status(out,'complete',rows=625,vote_minus_lock_pp=complete['vote_minus_lock_pp'],
            result_sha256=hashlib.sha256((target/'complete.json').read_bytes()).hexdigest(),next_iteration_authorized=False)
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_replay=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('tree','source','queue','out'):p.add_argument('--'+k,type=Path,required=True)
    run(**vars(p.parse_args()))
