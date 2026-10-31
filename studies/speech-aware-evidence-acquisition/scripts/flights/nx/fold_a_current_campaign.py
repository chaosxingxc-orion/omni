"""Execute one admitted full Fold A campaign through the shared flight runtime.

The caller detaches this coordinator. No stage retries a spent identity. Full
branch diagnostics and system-gain analysis require separate registered READs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
for directory in (Path(__file__).parent,REPO/'scripts/analysis'):
    sys.path.insert(0,str(directory))
import fold_a_current_registration as registration
import agent_port_score_evidence as delivered
import agent_port_fit_evidence as fit
from speech_aware_evidence_acquisition.core.atomic_file import publish_new


def validate(repo,data):
    v=registration.evaluation
    exp=registration.read(repo/f'configs/experiment/{v.CELL}.json');cell,=exp['cells']
    base=registration.read(repo/f'configs/baseline/{v.CELL}.json')
    if (exp['experiment_id']!=v.RUN or cell['cell_id']!=v.CELL or cell['baseline_fragment']!=v.CELL
            or (cell['run_id'],cell['attempt_id'])!=(v.RUN,'generation-v1')
            or (base['native_contact_fit']['run_id'],base['native_contact_fit']['attempt_id'])!=(v.FIT,'fit-v1')
            or cell['readiness']['state']!='ready' or 'campaign' not in cell['allowed_execution_stages']
            or base['policy']!='phase-native-module-v1-c17' or exp['family_budget']['official_complete_scoring_jobs']!=1):
        raise ValueError('ready current A0 Fold A campaign required')
    roster=registration.read(repo/v.comparator.inputs.ROSTER)
    ids=[r['sample_id'] for r in roster['items']]
    if (len(ids)!=1080 or len(set(ids))!=1080 or exp['populations'][0]['ids']!=ids
            or exp['population']['count']!=1080 or exp['population']['split_identity_hash']!=roster['split_identity_hash']):
        raise ValueError('exact full Fold A roster required')
    for run in (v.RUN,v.FIT):
        if any((data/n/run).exists() for n in ('runs','attempts')):raise ValueError('attempt already started; inspect without replay')
    if (data/'prepared'/v.FIT).exists():raise ValueError('FIT already started; inspect without replay')
    pin=base['prepared_phase_supply'];supply=v.comparator.inputs.confined(data,pin['relpath'])
    if not supply.is_relative_to((data/'prepared'/v.RUN).resolve()) or registration.sha(supply)!=pin['sha256']:
        raise ValueError('own Fold A supply differs')
    decision=data/v.DECISION_PATH;receipt=registration.read(decision)
    if (receipt['status']!='decided' or receipt['decision_id']!=v.DECISION
            or receipt['eligible_run_ids']!=[v.RUN,v.FIT] or receipt['eligible_attempt_ids']!=['generation-v1','fit-v1']):
        raise ValueError('exact Fold A admission required')
    recipe=repo/registration.PKG/'analysis-recipe-v1.json'
    delivered.require_read(repo,recipe,registration.READ,v.CELL)
    binding=receipt['budget']['accelerator_bound']['measurement']
    registration.validate_binding(binding,repo,data);registration.accepted_proof(repo,data,binding)
    if exp['service_session']!=binding['session_id']:raise ValueError('accepted serving identity differs')
    return {'session':exp['service_session'],'recipe':recipe,'supply':supply,'decision':decision,'rows':1080}


def commands(repo,data,lock,plan):
    v=registration.evaluation
    common=['--repo-root',str(repo),'--data-root',str(data),'--runtime-root',str(data),'--lock-path',str(lock),
            '--experiment',v.CELL,'--cell',v.CELL]
    cli=[sys.executable,'-m','speech_aware_evidence_acquisition.core.flight.cli']
    return [('registration',[sys.executable,'scripts/checks/flight_registration_check.py','--repo-root',str(repo),
                              '--data-root',str(data),'--cell',v.CELL,'--json']),
            ('preflight',cli+['check','--read-only',*common]),
            ('campaign',cli+['campaign',*common,'--owner','root-current-a0-fold-a','--expected-session-id',plan['session']])]


def execute(repo,data,lock,out,source_commit):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    repo,data=repo.resolve(),data.resolve()
    out=registration.evaluation.comparator.inputs.confined(data,str(out))
    if out.exists() or out.is_relative_to(repo):raise ValueError('fresh external coordinator required')
    def source_unchanged():
        if (subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
                or subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=source_commit):
            raise ValueError('pinned clean source required')
    source_unchanged()
    if (data/'locks/flight.lock').exists():raise ValueError('generation lease held; wait for release')
    plan=validate(repo,data)
    pins={str(p):registration.sha(p) for p in [plan['recipe'],plan['supply'],plan['decision'],lock]}
    out.mkdir(parents=True,exist_ok=False)
    def publish(name,value):publish_new(out/name,registration.raw(value))
    publish('started.json',{'pid':os.getpid(),'commit':source_commit,'run_id':registration.evaluation.RUN,
        'time':datetime.now(timezone.utc).isoformat(),'inputs':pins,'rows':1080})
    try:
        for name,args in commands(repo,data,lock,plan):
            source_unchanged()
            if any(registration.sha(Path(p))!=h for p,h in pins.items()):raise ValueError('pinned input changed')
            if name=='campaign':
                validate(repo,data)
                if (data/'locks/flight.lock').exists():raise ValueError('generation lease acquired during checks')
            publish(name+'.intent.json',{'argv':args,'commit':source_commit,'read_id':registration.READ})
            with (out/(name+'.log')).open('x') as log:
                result=subprocess.run(args,cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'SAEA_ROLLOUT_TRACE_READ_ID':registration.READ})
            publish(name+'.exit.json',{'exit':result.returncode,'time':datetime.now(timezone.utc).isoformat()})
            if result.returncode:raise RuntimeError(name+' failed; inspect existing evidence without replay')
        source_unchanged()
        if any(registration.sha(Path(p))!=h for p,h in pins.items()):raise ValueError('pinned input changed during campaign')
        publish('native-fit-evidence.json',fit.build(repo,data,registration.evaluation.CELL,registration.READ))
        final=delivered.build(repo,data,registration.evaluation.CELL,registration.READ,plan['recipe'])
        publish('final-delivered-score-evidence.json',final)
        record={'commit':source_commit,'run_id':registration.evaluation.RUN,'rows':final['rows'],
                'aggregate':final['aggregate'],'score_surface':'sealed-final-delivery','claims_system_benefit':False,
                'remaining':['registered branch diagnostics and paired analysis','initial Direct/A0 system lock']}
        publish('complete.json',record)
        return record
    except BaseException as error:
        publish('failed.json',{'error':str(error),'automatic_retry':False});raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','data-root','lock-path','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);a=p.parse_args()
    print(json.dumps(execute(a.repo_root,a.data_root,a.lock_path,a.out,a.source_commit),sort_keys=True))


if __name__=='__main__':main()
