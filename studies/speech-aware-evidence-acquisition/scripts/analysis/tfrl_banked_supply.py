"""Prepare each real banked half's own first-phase supply through FlightRuntime.

No generation, FIT, tokenizer, scorer or readiness promotion. Source bank and
first construction decision must already exist and pass the shared real guard.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
from tfrl_current_generation import read,sha,validate_half,LOCK,LOCK_SHA
from speech_aware_evidence_acquisition.core.atomic_file import publish_new


def validate_preparation_inputs(repo,data,name,iteration,first_decision,*,allow_bound=False):
    if type(iteration) is not int or iteration not in (2,3):
        raise ValueError('banked iteration2/3 required')
    ep=repo/f'configs/experiment/{name}.json';exp=read(ep);c,=exp['cells']
    bp=repo/f"configs/baseline/{c['baseline_fragment']}.json";base=read(bp)
    if (c['cell_id']!=name or c['baseline_fragment']!=name
            or c['readiness']['state']!='held' or c['allowed_execution_stages']!=['check']
            or (not allow_bound and 'prepared_phase_supply' in base)):
        raise ValueError('unbound held banked half required')
    run=c['run_id'];fit=base['native_contact_fit']
    for rid in (run,fit['run_id']):
        if any((data/k/rid).exists() for k in ('runs','attempts')):
            raise ValueError('existing generation or FIT spend; inspect before continuation')
    def pin(path,root=repo):
        return {'relpath':str(path.resolve().relative_to(root.resolve())),'sha256':sha(path)}
    dp=repo/f"configs/dataset/{c['dataset_fragment']}.json"
    pp=repo/f'docs/readiness/{name}/sampling-plan.json'
    rp=repo/exp['population']['roster_receipt']
    decision_pin=pin(first_decision,data)
    spec={'run_id':run,'attempt_id':c['attempt_id'],'rows':exp['population']['count'],
        'split_identity_hash':exp['population']['split_identity_hash'],'iteration':iteration,
        'gain_lock':{'relpath':LOCK,'sha256':LOCK_SHA},'dataset':pin(dp),
        'sampling_plan':pin(pp),'roster':pin(rp),'first_construction_decision':decision_pin,
        'fit_run_id':fit['run_id'],'fit_attempt_id':fit['attempt_id']}
    rows=validate_half(repo,name,spec,exp,read(dp),data=data,iteration=iteration)
    pins={str(p.relative_to(repo)):sha(p) for p in (ep,bp,dp,pp,rp,repo/LOCK)}
    return exp,rows,pins,decision_pin


def prepare(repo,data,name,iteration,first_decision,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment();head=require_clean_study_repo(repo)
    exp,rows,pins,decision_pin=validate_preparation_inputs(repo,data,name,iteration,first_decision)
    c,=exp['cells'];run=c['run_id'];target=data/'prepared'/run/'phase-supply-v1.json'
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex') or target.parent.exists():
        raise ValueError('fresh owned supply required; preserve existing preparation')
    def write(path,doc):publish_new(path,(json.dumps(doc,indent=2)+'\n').encode())
    out.mkdir(parents=True)
    write(out/'intent.json',{'source_commit':head,'rows':rows,'source_hashes':pins,
        'first_construction_decision':decision_pin,'model_calls':0,'automatic_replay':False})
    try:
        runtime=build_flight_runtime(repo,data,model=c['model_fragment'],dataset=c['dataset_fragment'],
            baseline=c['baseline_fragment'],experiment=name,run_id=run,attempt_id=c['attempt_id'],
            runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json')
        runtime.prepare_phase_supply(target,max_workers=16)
        output=read(target)['output'];ids=set(exp['populations'][0]['ids'])
        if not output or len(ids)!=rows:
            raise ValueError('empty or incomplete prepared supply')
        for phases in output.values():
            if not phases:raise ValueError('empty supply phases')
            for entries in phases.values():
                if set(entries)!=ids:raise ValueError('supply is not the exact real half')
        # Refuse publication if source or measured bank changed during CPU work.
        if require_clean_study_repo(repo)!=head:
            raise ValueError('preparation source commit changed')
        _,_,current,decision=validate_preparation_inputs(repo,data,name,iteration,first_decision)
        if current!=pins or decision!=decision_pin:
            raise ValueError('preparation inputs changed')
        result={'schema':'saea-tfrl-current-supply-v1','source_commit':head,'cell':name,
            'run_id':run,'iteration':iteration,'rows':rows,'source_hashes':pins,
            'first_construction_decision':decision_pin,
            'supply':{'relpath':str(target.relative_to(data)),'sha256':sha(target)},
            'model_calls':0,'tokenizer_calls':0,'scoring_jobs':0,'readiness':'held','registered':False}
        write(out/'complete.json',result);return result
    except BaseException as error:
        write(out/'failed.json',{'error':str(error),'automatic_replay':False,
            'preserve':'prepared supply and intent; inspect before continuation'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','first-decision','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--iteration',type=int,required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
