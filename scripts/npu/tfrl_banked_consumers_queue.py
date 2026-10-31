"""Detached later-iteration CPU consumers;600s polling, no blind replay.

Launch with nohup setsid on npu2. Does not start generation or change readiness.
Complete source receipts are revalidated by the formal registered consumer.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from real500_successor_queue import D,IMAGE,write,status


def identities(iteration):
    if type(iteration) is not int or iteration not in (2,3):raise ValueError('banked iteration2/3 required')
    prefix=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}'
    return prefix+'-POOL-READ-V1',prefix+'-BANK-V1',f'docs/readiness/tfrl-current-bank-i{iteration}-consumers/analysis-recipe-v1.json'


def ready(results,iteration):
    identities(iteration)
    if set(results)!={'h1','h2'}:raise ValueError('two source halves required')
    for half,result in results.items():
        if result is None:continue
        expected={'run_id':f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{half.upper()}-V1',
            'attempt_id':'generation-v1','iteration':iteration,'rows':313 if half=='h1' else 312,
            'official_scoring_jobs':7,'new_model_calls':0}
        if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('upstream result identity differs')
    return all(r is not None for r in results.values())


def require_live_upstream(document,kill=os.kill):
    if document.get('state') in ('failed','complete'):
        raise ValueError('upstream terminal without complete result; inspect without replay')
    pid=document.get('pid')
    if type(pid) is not int or pid<=1:raise ValueError('upstream has no process identity')
    try:kill(pid,0)
    except ProcessLookupError as error:raise ValueError('upstream process missing without result') from error


def run(tree,queue_h1,queue_h2,out,iteration):
    pool,bank,rel=identities(iteration)
    study=tree/'studies/speech-aware-evidence-acquisition';recipe=study/rel
    read=lambda p:json.loads(p.read_bytes()) if p.exists() else None
    recipe_doc=read(recipe)
    if not recipe_doc or recipe_doc.get('iteration')!=iteration or recipe_doc.get('bank_run_id')!=bank:
        raise ValueError('registered later consumer recipe required')
    sources=recipe_doc['sources']
    if set(sources)!={'h1','h2'}:raise ValueError('two pinned sources required')
    roots=[(D/sources[h]['repo_relpath']).resolve() for h in ('h1','h2')]
    if not all(p.is_relative_to(D) for p in roots):raise ValueError('source escapes data root')
    def git(*args):return subprocess.check_output(['git','-C',str(study),*args],text=True).strip()
    commit=git('rev-parse','HEAD');digest=hashlib.sha256(recipe.read_bytes()).hexdigest()
    def check():
        if git('rev-parse','HEAD')!=commit or git('status','--porcelain') or hashlib.sha256(recipe.read_bytes()).hexdigest()!=digest:
            raise ValueError('consumer implementation or recipe changed')
    check()
    if out.exists() or not out.resolve().is_relative_to(D/'work/codex'):
        raise ValueError('fresh owned queue required')
    if any((D/'analysis'/rid).exists() for rid in (pool,bank)):
        raise ValueError('existing consumer output requires explicit continuation')
    out.mkdir(parents=True)
    write(out/'intent.json',{'iteration':iteration,'source_commit':commit,'recipe_sha256':digest,
        'interval_seconds':600,'generation_calls':0,'scoring_jobs':0,'automatic_replay':False})
    queues={'h1':queue_h1,'h2':queue_h2}
    targets={h:D/'analysis'/f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{h.upper()}-V1-READ/complete.json' for h in queues}
    try:
        status(out,'waiting_complete_official_halves',iteration=iteration)
        while True:
            results={h:read(p) for h,p in targets.items()}
            if ready(results,iteration):break
            for h,result in results.items():
                if result is None:require_live_upstream(read(queues[h]/'status.json') or {})
            time.sleep(600)
        check()
        # Exact safe directories only; never broaden ownership acceptance globally.
        safe=list(dict.fromkeys([git('rev-parse','--show-toplevel'),*[str(p) for p in roots]]))
        flags=['--network','host','--name',f'saea-tfrl-banked-i{iteration}-consumers',
            '-e',f'GIT_CONFIG_COUNT={len(safe)}']
        for index,path in enumerate(safe):
            if any(c.isspace() for c in path):raise ValueError('safe directory with whitespace unsupported by host wrapper')
            flags+=['-e',f'GIT_CONFIG_KEY_{index}=safe.directory','-e',f'GIT_CONFIG_VALUE_{index}={path}']
        env={**os.environ,'TREE':str(tree),'CPUSET':'224-239','PORT_CONTAINER_USER':'0:0',
            'PORT_IMAGE':IMAGE,'PORT_DOCKER_FLAGS':' '.join(flags)}
        command=['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/tfrl_recovered_consumers.py',
            'bank','--repo',str(study),'--data',str(D),'--iteration',str(iteration)]
        with (out/'consumers.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            status(out,'running_registered_consumers',child_pid=child.pid,iteration=iteration)
            code=child.wait()
        write(out/'exit.json',{'exit':code,'time':time.time()})
        if code:raise ValueError('consumer failed; retain tokenizer journal without replay')
        completion=read(D/'analysis'/bank/'complete.json')
        if (not completion or completion['rows']!=625 or completion['recipe_sha256']!=digest
                or completion['accounting']['generation_calls']!=0
                or completion['accounting']['unresolved_tokenizer_requests']):
            raise ValueError('bank completion differs')
        status(out,'complete',iteration=iteration,rows=625,next_iteration_authorized=False)
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_replay=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('tree','queue-h1','queue-h2','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True)
    run(**vars(p.parse_args()))
