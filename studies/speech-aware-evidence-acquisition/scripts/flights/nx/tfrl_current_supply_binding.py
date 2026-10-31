"""Bind completed real K4 half supplies without admission or new model calls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
import r01c_current_baseline as baseline
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read,sha,raw=baseline.read,baseline.sha,baseline.raw


def bound_documents(base, exp, prep, *, iteration=1):
    if type(iteration) is not int or iteration not in (1,2,3):
        raise ValueError("iteration1..3 required")
    prefix = "seed" if iteration==1 else "bank"
    cell,=exp['cells'];n=exp['population']['count'];name=cell['cell_id'];run=cell['run_id']
    half = 'h1' if n == 313 else 'h2' if n == 312 else None
    if (half is None or name != f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1'
            or run != f'SAEA-SLURP-TFRL-CURRENT-{prefix.upper()}-I{iteration}-{half.upper()}-V1'
            or prep['schema']!='saea-tfrl-current-supply-v1' or prep['rows']!=n
            or prep['cell']!=name or prep['run_id']!=run or prep['model_calls']!=0
            or prep['readiness']!='held' or prep['registered'] is not False
            or cell['readiness']['state']!='held' or cell['allowed_execution_stages']!=['check']
            or 'prepared_phase_supply' in base):
        raise ValueError('unbound held real half and exact completed supply required')
    if iteration>1 and prep.get('iteration')!=iteration:
        raise ValueError('banked supply iteration differs')
    if prep['supply']['relpath']!=f'prepared/{run}/phase-supply-v1.json':
        raise ValueError('own half supply required')
    base,exp=copy.deepcopy(base),copy.deepcopy(exp)
    base['prepared_phase_supply']=prep['supply']
    exp['cells'][0]['readiness']['requirements'].append({'kind':'preparation',**prep['supply']})
    return base,exp


def bind(repo,data,preparations,out,iteration=1):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment();head=require_clean_study_repo(repo)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned binding required')
    if len(preparations)!=2:raise ValueError('both completed halves required')
    prefix=subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-prefix'],text=True).strip()
    docs=[read(p) for p in preparations]
    if {d['rows'] for d in docs}!={313,312}:raise ValueError('exact two halves required')
    pending=[]
    for prep in docs:
        run=prep['run_id'];name=prep['cell']
        if any((data/k/run).exists() for k in ('runs','attempts')):raise ValueError('half already spent')
        target=baseline.comparator.confined(data,prep['supply']['relpath'])
        if sha(target)!=prep['supply']['sha256']:raise ValueError('supply changed')
        for rel,digest in prep['source_hashes'].items():
            source=baseline.comparator.confined(repo,rel)
            if sha(source)!=digest:raise ValueError('prepared source changed')
            previous=subprocess.check_output(['git','-c','safe.directory='+str(repo),'-C',str(repo),
                'show',prep['source_commit']+':'+prefix+rel])
            if hashlib.sha256(previous).hexdigest()!=digest:raise ValueError('preparation commit differs')
        bp=repo/f'configs/baseline/{name}.json';ep=repo/f'configs/experiment/{name}.json'
        base,exp=bound_documents(read(bp),read(ep),prep,iteration=iteration)
        if iteration>1:
            sys.path.insert(0,str(repo/'scripts/analysis'))
            from tfrl_banked_supply import validate_preparation_inputs
            _,_,expected_sources,_=validate_preparation_inputs(repo,data,name,iteration,
                data/prep['first_construction_decision']['relpath'])
            if prep['source_hashes']!=expected_sources:
                raise ValueError('complete preparation source pins required')
            pin=prep['first_construction_decision']
            if sha(baseline.comparator.confined(data,pin['relpath']))!=pin['sha256']:
                raise ValueError('first construction decision changed after preparation')
        ids=set(exp['populations'][0]['ids']);outputs=read(target)['output']
        if not outputs or len(ids)!=prep['rows']:raise ValueError('empty or incomplete supply')
        for kinds in outputs.values():
            for rows in kinds.values():
                if set(rows)!=ids:raise ValueError('half supply row mismatch')
        pending.append((prep,bp,ep,base,exp))
    out.mkdir(parents=True)
    publish_new(out/'intent.json',raw({'source_commit':head,'preparations':{str(p):sha(p) for p in preparations},'new_model_calls':0}))
    for prep,bp,ep,base,exp in pending:
        for p in (bp,ep):publish_new(out/(p.stem+'.'+p.parent.name+'.before.json'),p.read_bytes())
        bp.write_bytes(raw(base));ep.write_bytes(raw(exp))
    checks=[];pins={}
    for prep,bp,ep,base,exp in pending:
        c,=exp['cells']
        runtime=build_flight_runtime(repo,data,model=c['model_fragment'],dataset=c['dataset_fragment'],
            baseline=c['baseline_fragment'],experiment=c['cell_id'],run_id=c['run_id'],attempt_id=c['attempt_id'],
            runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json')
        verified=runtime.verify_prepared_artifacts()
        if not verified or not all(v['identity_ok'] and v.get('seed_ok',False) for v in verified):
            raise ValueError('shared prepared identity or seed verification failed')
        checks.append({'cell':c['cell_id'],'rows':prep['rows'],'checks':verified,'supply':prep['supply']})
        pins.update({str(p.relative_to(repo)):sha(p) for p in (bp,ep)})
    result={'schema':'saea-tfrl-current-supply-binding-v1','rows':625,'halves':checks,'output_hashes':pins,
        'source_commit':head,'new_model_calls':0,'readiness':'held','registered':False}
    publish_new(out/'complete.json',raw(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--preparation',dest='preparations',action='append',type=Path,required=True)
    p.add_argument('--iteration',type=int,default=1)
    print(json.dumps(bind(**vars(p.parse_args()))))
