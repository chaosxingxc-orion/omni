"""Execute the two admitted real K4 halves; overlap CPU reads with next inference.

Launch once detached. Verified lease release is the handoff signal; no model polling.
Existing attempts and partial analysis are preserved and never blindly replayed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from generation_release_handoff import wait_release
from real500_successor_queue import D, IMAGE, write, status


def run_lanes(jobs, launch_generation, launch_read, record, await_release=None):
    """A failed CPU read does not prevent an independent authorised generation."""
    if await_release is not None:
        failure = None
        def finalize(job, child):
            code = child.wait()
            record('generation_exit', job, exit=code)
            if code:
                raise RuntimeError('finalization failed after release; preserve generation: ' + job)
            code = launch_read(job).wait()
            record('read_exit', job, exit=code)
            if code:
                raise RuntimeError('official read failed; preserve scoring receipts: ' + job)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []
            try:
                for job in jobs:
                    record('generating', job)
                    child = launch_generation(job)
                    await_release(job, child)
                    record('generation_released', job)
                    futures.append(pool.submit(finalize, job, child))
            except BaseException as error:
                failure = error
            for future in futures:
                try:
                    future.result()
                except BaseException as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure
        return
    readers = []
    failure = None
    try:
        for job in jobs:
            record('generating', job)
            code = launch_generation(job).wait()
            record('generation_exit', job, exit=code)
            if code:
                raise RuntimeError('generation failed; preserve checkpoints: ' + job)
            readers.append((job, launch_read(job)))
    except BaseException as error:
        failure = error
    for job, child in readers:
        code = child.wait()
        record('read_exit', job, exit=code)
        if code and failure is None:
            failure = RuntimeError('official read failed; preserve scoring receipts: ' + job)
    if failure is not None:
        raise failure


def run(tree, out, iteration=1):
    if type(iteration) is not int or iteration not in (1,2,3):
        raise ValueError("iteration1..3 required")
    prefix = "seed" if iteration == 1 else "bank"
    if not out.resolve().is_relative_to(D/'work/codex') or out.exists():
        raise ValueError('fresh owned queue required; inspect existing handles before any continuation')
    study = tree/'studies/speech-aware-evidence-acquisition'
    head = subprocess.check_output(['git','-C',str(study),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(study),'status','--porcelain']).strip():
        raise ValueError('clean admitted execution tree required')
    jobs = ('h1','h2')
    specs = {}
    for half in jobs:
        name = f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1'
        spec = json.loads((study/'docs/readiness'/name/'candidate.json').read_bytes())
        expected = {'run_id':f'SAEA-SLURP-TFRL-CURRENT-{prefix.upper()}-I{iteration}-{half.upper()}-V1',
                    'rows':313 if half=='h1' else 312,
                    'workload_kind':'tfrl-real-seed-half' if iteration==1 else 'tfrl-real-banked-half',
                    'profile':'qwen3-omni-real500-2xtp4-s128-b2048-v1'}
        if iteration>1:expected['iteration']=iteration
        if any(spec.get(k)!=v for k,v in expected.items()):
            raise ValueError('queue candidate identity differs')
        recipe=json.loads((study/'docs/readiness'/name/'analysis-recipe-v1.json').read_bytes())
        if any(recipe.get(k)!=spec[k] for k in ('run_id','attempt_id','rows')):
            raise ValueError('registered official read identity differs')
        specs[half] = spec
        exp = json.loads((study/'configs/experiment'/f'{name}.json').read_bytes())
        if exp['cells'][0]['readiness']['state'] != 'ready':
            raise ValueError('both halves must be admitted before queue launch')
        for rid in (spec['run_id'],spec['fit_run_id']):
            if any((D/k/rid).exists() for k in ('runs','attempts')):
                raise ValueError('existing spend requires explicit checkpoint continuation')
        if (D/'analysis'/(spec['run_id']+'-READ')).exists():
            raise ValueError('existing read must not be replayed')
    out.mkdir(parents=True)
    write(out/'intent.json',{'pid':os.getpid(),'source_commit':head,'tree':str(tree),
        'rows':625,'halves':[313,312],'resources':{'cards':8,'instances':2,'tp':4,'max_num_seqs':128},
        'monitor_interval_seconds':600,'normal_model_calls':4375,'analysis_overlap':True,
        'iteration':iteration,'handoff':'verified_generation_lease_release'})
    record_lock = threading.Lock()
    def record(event,half,**extra):
        with record_lock:
            write(out/(half+'-'+event+'.json'),{'event':event,'half':half,'time':time.time(),**extra})
            status(out,event,half=half,**extra)
    def launch(half,read=False):
        current = subprocess.check_output(['git','-C',str(study),'rev-parse','HEAD'],text=True).strip()
        if current != head or subprocess.check_output(['git','-C',str(study),'status','--porcelain']).strip():
            raise ValueError('execution tree changed during queue')
        name=f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1';run_id=specs[half]['run_id']
        mode='read' if read else 'generation'
        env={**os.environ,'TREE':str(tree),'CPUSET':'208-223' if read else '160-191',
             'PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,
             'SAEA_DEPLOYMENT_PROFILE':'qwen3-omni-real500-2xtp4-s128-b2048-v1',
             'PORT_DOCKER_FLAGS':('--network none' if read else '--pid host --network host')+
                 f' --name saea-tfrl-current-i{iteration}-{half}-{mode}-v1'+
                 ('' if read else f' -e SAEA_STAGE_TIMING_PATH={out}/{half}-stages.jsonl')}
        command=['bash',str(D/'work/scripts/port_py.sh')]
        if read:
            command+=['scripts/analysis/tfrl_current_half_result.py','--repo',str(study),'--data',str(D),
                '--recipe-path',str(study/'docs/readiness'/name/'analysis-recipe-v1.json'),
                '--out',str(D/'analysis'/(run_id+'-READ'))]
        else:
            command+=['scripts/analysis/run_real_engine_candidate.py','--repo',str(study),'--data',str(D),
                '--candidate',name,'--out',str(out/(half+'-run'))]
        with (out/(half+'-'+mode+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        write(out/(half+'-'+mode+'-process.json'),{'pid':child.pid,'command':command,'time':time.time()})
        return child
    def released(half, child):
        spec = specs[half]
        expected = {k: spec[k] for k in ('run_id','attempt_id','profile','service_session','rows')}
        expected.update(candidate=f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1',study_commit=head)
        return wait_release(child,out/(half+'-run')/'generation-released.json',data=D,expected=expected)
    try:
        run_lanes(jobs,lambda h:launch(h),lambda h:launch(h,True),record,released)
        status(out,'complete',rows=625,official_jobs=14,new_generation_done=True)
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_replay=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('tree','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=1)
    run(**vars(p.parse_args()))
