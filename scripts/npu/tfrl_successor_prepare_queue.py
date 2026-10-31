"""Wait for real625, verify the kill rule, then stage held successor configs.

Run detached on npu2. No generation, tokenizer, scoring, or admission changes.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from real500_successor_queue import D,IMAGE,write,status
from tfrl_banked_consumers_queue import require_live_upstream
POOL='SAEA-SLURP-TFRL-CURRENT-SEED-I1-RECOVERED-POOL-READ-V1'
BANK='SAEA-SLURP-TFRL-CURRENT-SEED-I1-RECOVERED-BANK-V1'


def decision_state(document):
    if (document.get('schema')!='saea-tfrl-first-construction-decision-v1'
            or document.get('rows')!=625 or document.get('generation_authorized') is not False
            or document.get('system_accepted') is not False):
        raise ValueError('unexpected verified decision')
    if document.get('decision')=='stop-family' and document.get('preparation_gate_passed') is False:return 'stopped-family'
    if document.get('decision')=='prepare-held-successor' and document.get('preparation_gate_passed') is True:return 'prepare'
    raise ValueError('inconsistent verified decision')


def run(tree,upstream,out):
    if out.exists() or not out.resolve().is_relative_to(D/'work/codex'):raise ValueError('fresh owned preparation queue required')
    study=tree/'studies/speech-aware-evidence-acquisition'
    def git(*args):return subprocess.check_output(['git','-C',str(study),*args],text=True).strip()
    commit=git('rev-parse','HEAD')
    def check():
        if git('status','--porcelain') or git('rev-parse','HEAD')!=commit:raise ValueError('preparation source changed')
    check()
    read=lambda p:json.loads(p.read_bytes()) if p.exists() else None
    recipe=read(study/'docs/readiness/tfrl-current-recovered-consumers/analysis-recipe-v1.json')
    roots=[(D/s['repo_relpath']).resolve() for s in recipe['sources'].values()]
    if not all(p.is_relative_to(D) for p in roots):raise ValueError('source outside data root')
    safe=list(dict.fromkeys([git('rev-parse','--show-toplevel'),*[str(p) for p in roots]]))
    flags=['--network','none','--name','saea-tfrl-successor-held-preparation-v1','-e',f'GIT_CONFIG_COUNT={len(safe)}']
    for i,path in enumerate(safe):
        if any(c.isspace() for c in path):raise ValueError('unsupported whitespace in directory')
        flags+=['-e',f'GIT_CONFIG_KEY_{i}=safe.directory','-e',f'GIT_CONFIG_VALUE_{i}={path}']
    env={**os.environ,'TREE':str(tree),'CPUSET':'192-207','PORT_CONTAINER_USER':'0:0',
         'PORT_IMAGE':IMAGE,'PORT_DOCKER_FLAGS':' '.join(flags)}
    out.mkdir(parents=True)
    write(out/'intent.json',{'implementation_commit':commit,'interval_seconds':600,'generation_calls':0,
        'tokenizer_calls':0,'scoring_jobs':0,'automatic_replay':False})
    def wait_for(target,phase):
        status(out,phase)
        while not target.exists():
            require_live_upstream(read(upstream/'status.json') or {})
            time.sleep(600)
        check()
    def execute(name,target,decision=False):
        command=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/tfrl_current_next_plan.py',
            '--repo',str(study),'--data',str(D),'--out',str(target)]
        if decision:command+=['--decision-only']
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            status(out,name,child_pid=child.pid);code=child.wait()
        write(out/(name+'.exit.json'),{'exit':code})
        if code:raise ValueError(name+' failed; preserve artifacts without replay')
    try:
        wait_for(D/'analysis'/POOL/'complete.json','waiting_verified_full625')
        execute('decision',out/'decision.json',True)
        decision=read(out/'decision.json');state=decision_state(decision)
        if state=='stopped-family':
            status(out,state,construction_gain_pp=decision['construction_gain_pp'],generation_authorized=False);return
        wait_for(D/'analysis'/BANK/'complete.json','waiting_measured_banks')
        execute('held-staging',out/'held-successor')
        complete=read(out/'held-successor/complete.json')
        if (complete.get('rows')!=625 or complete.get('iteration')!=2 or complete.get('readiness')!='held'
                or complete.get('generation_authorized_by_this_file') is not False):
            raise ValueError('unexpected held successor completion')
        status(out,'held-successor-prepared',rows=625,generation_authorized=False,
            complete_sha256=hashlib.sha256((out/'held-successor/complete.json').read_bytes()).hexdigest())
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_replay=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('tree','upstream','out'):p.add_argument('--'+k,type=Path,required=True)
    run(**vars(p.parse_args()))
