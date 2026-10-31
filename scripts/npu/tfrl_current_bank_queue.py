"""Run the registered experience producer once after both complete half reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D,IMAGE,write,status
from tfrl_current_pool_queue import ready


def run(tree,source,queue,out):
    if out.exists() or not out.resolve().is_relative_to(D/'work/codex'):raise ValueError('fresh owned bank queue required')
    study=tree/'studies/speech-aware-evidence-acquisition';src=source/'studies/speech-aware-evidence-acquisition'
    recipe=study/'docs/readiness/tfrl-current-seed-i1-bank/analysis-recipe-v1.json'
    digest=hashlib.sha256(recipe.read_bytes()).hexdigest()
    commit=subprocess.check_output(['git','-C',str(study),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(study),'status','--porcelain']).strip():raise ValueError('clean bank source required')
    target=D/'analysis/SAEA-SLURP-TFRL-CURRENT-SEED-I1-BANK-V1'
    if target.exists():raise ValueError('existing bank attempt requires explicit journal continuation')
    out.mkdir(parents=True);write(out/'intent.json',{'pid':os.getpid(),'source_commit':commit,'recipe_sha256':digest,
        'generation_calls':0,'official_scoring_jobs':0,'tokenizer_contact_cap':20000,'interval_seconds':600})
    read=lambda p:json.loads(p.read_bytes()) if p.exists() else None
    status(out,'waiting_two_complete_halves')
    try:
        while not ready([read(D/'analysis'/f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{h}-V1-READ/complete.json') for h in ('H1','H2')],read(queue/'status.json') or {}):
            time.sleep(600)
        if (hashlib.sha256(recipe.read_bytes()).hexdigest()!=digest
                or subprocess.check_output(['git','-C',str(study),'rev-parse','HEAD'],text=True).strip()!=commit
                or subprocess.check_output(['git','-C',str(study),'status','--porcelain']).strip()):
            raise ValueError('bank producer source changed')
        status(out,'verifying_sources_and_distilling')
        env={**os.environ,'TREE':str(tree),'CPUSET':'192-207','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,
            'PORT_DOCKER_FLAGS':'--pid host --network host --name saea-tfrl-current-bank-production-v1'}
        command=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/tfrl_current_bank_run.py','run',
            '--repo',str(study),'--source',str(src),'--data',str(D)]
        with (out/'bank.log').open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
        write(out/'exit.json',{'exit':result.returncode,'time':time.time()})
        if result.returncode:raise ValueError('bank production failed; inspect durable token journal without blind replay')
        complete=read(target/'complete.json')
        if (complete['rows']!=625 or complete['claims_benefit'] is not False
                or complete['accounting']['generation_calls']!=0 or complete['accounting']['unresolved_tokenizer_requests']):
            raise ValueError('unexpected bank completion')
        status(out,'complete',rows=625,banks=complete['banks'],accounting=complete['accounting'],next_iteration_authorized=False)
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_replay=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('tree','source','queue','out'):p.add_argument('--'+k,type=Path,required=True)
    run(**vars(p.parse_args()))
