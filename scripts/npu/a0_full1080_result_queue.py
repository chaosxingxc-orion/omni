"""Score the source-bound full A0 roster once after missing353 finalizes.

Observe terminal files every600s. Never replay inference, score a partial union,
or retry a failed official scoring invocation automatically.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D,IMAGE,write,status

RUN='SAEA-SLURP-A0-REMAINING353-C2-V1'
ATTEMPT='attempt-a0-remaining353-c2-v1'
RECIPE='docs/readiness/2026-10-31-a0-full1080-source-read/analysis-recipe-v1.json'
RECIPE_SHA='efb43a80731ccc01dfb0fa038664687b188f416d56063e23628ae29937e0db38'


def check_terminal(handoff,terminal,manifest,attempt):
    if handoff.get('state')=='failed':raise ValueError('A0 handoff failed; preserve evidence')
    if terminal is None:return False
    if terminal.get('exit')!=0:raise ValueError('continuation invocation failed')
    if manifest is None or attempt is None:raise ValueError('successful invocation lacks sealed evidence')
    if (manifest.get('run_id')!=RUN or manifest.get('status')!='COMPLETED'
            or attempt.get('run_id')!=RUN or attempt.get('attempt_id')!=ATTEMPT
            or attempt.get('state')!='FINALIZED'):
        raise ValueError('continuation not finalized under expected identity')
    return True


def run(tree,out):
    if not out.resolve().is_relative_to(D/'work/codex') or out.exists():
        raise ValueError('fresh owned result queue required')
    study=tree/'studies/speech-aware-evidence-acquisition'
    files=[study/'scripts/analysis'/name for name in ('score_a0_full1080.py','prepare_a0_source_custody.py','audit_real500_a0_inputs.py')]+[study/RECIPE]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(study/RECIPE)!=RECIPE_SHA:raise ValueError('registered read recipe changed')
    pins={str(p):sha(p) for p in files}
    target=D/'analysis/SAEA-SLURP-A0-FULL1080-SOURCE-READ-V1'
    if target.exists():raise ValueError('analysis already attempted; inspect without replay')
    out.mkdir(parents=True)
    write(out/'intent.json',{'pid':os.getpid(),'tree':str(tree),'pins':pins,
        'interval_seconds':600,'new_model_calls':0,'official_scoring_jobs':1,'output':str(target)})
    handoff=D/'work/codex/a0-after-real500-queue-v1/handoff'
    terminal=handoff/'run.exit.json'
    manifest=D/'runs'/RUN/ATTEMPT/(RUN+'.run-manifest.json')
    attempt=D/'attempts'/RUN/(ATTEMPT+'.json')
    read=lambda p:json.loads(p.read_bytes()) if p.exists() else None
    status(out,'waiting_finalized353')
    try:
        while not check_terminal(read(handoff/'status.json') or {},read(terminal),read(manifest),read(attempt)):
            parent=read(handoff.parent/'status.json') or {}
            if parent.get('state')=='failed':raise ValueError('upstream A0 queue failed')
            time.sleep(600)
        if any(sha(Path(p))!=digest for p,digest in pins.items()):raise ValueError('scoring source/recipe changed')
        status(out,'verifying_and_scoring_full1080')
        reference=D/'work/dev/real500-runtime/studies/speech-aware-evidence-acquisition'
        env={**os.environ,'TREE':str(tree),'CPUSET':'160-175','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,
            'PORT_DOCKER_FLAGS':'--network none --name saea-a0-full1080-official-read-v1 '
                '-e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='+str(reference)}
        command=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/score_a0_full1080.py',
            '--repo',str(study),'--data',str(D),'--recipe-path',str(study/RECIPE),
            '--continuation-repo',str(study),'--continuation-manifest',str(manifest),'--out',str(target)]
        with (out/'score.log').open('x') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(out/'formal.exit.json',{'exit':result.returncode,'time':time.time()})
        if result.returncode:raise ValueError('full1080 read failed; preserve partial scoring evidence')
        complete=read(target/'complete.json');scores=read(target/'scores.json')
        if (complete['rows']!=1080 or complete['official_scoring_jobs']!=1
                or complete['scores_sha256']!=sha(target/'scores.json')
                or scores['aggregate']['samples']!=1080 or not scores['aggregate']['identity_check']['passed']):
            raise ValueError('full1080 completion differs')
        status(out,'complete',rows=1080,quality=scores['aggregate']['official'],
            complete_sha256=sha(target/'complete.json'),new_model_calls=0,
            next_required='Verify matched Direct comparison and lock first-system gain before TFRL.')
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_retry=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(**vars(p.parse_args()))
