"""Prepare the admitted Fold A evaluation's own phase supply, model-free."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import fold_a_current_registration as registration
from refresh_registered_plan_pins import refresh_package
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.flight.repin import repin_package,rewrite_dependent_hashes


def prepare(repo,data,lock,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    v=registration.evaluation
    out=v.comparator.inputs.confined(data,str(out))
    if out.exists() or out.is_relative_to(repo.resolve()):raise ValueError('fresh external preparation required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('clean registration required')
    decision=data/v.DECISION_PATH;before=decision.read_bytes();receipt=json.loads(before)
    if (receipt['status']!='decided' or receipt['decision_id']!=v.DECISION
            or receipt['eligible_run_ids']!=[v.RUN,v.FIT]):raise ValueError('minted Fold A admission required')
    exp=registration.read(repo/f'configs/experiment/{v.CELL}.json')
    base=registration.read(repo/f'configs/baseline/{v.CELL}.json')
    if exp['cells'][0]['readiness']['state']!='held' or base.get('prepared_phase_supply'):
        raise ValueError('unprepared held Fold A required')
    for run in (v.RUN,v.FIT):
        if any((data/name/run).exists() for name in ('prepared','runs','attempts')):raise ValueError('Fold A preparation or contact already started')
    out.mkdir(parents=True,exist_ok=False)
    publish_new(out/'intent.json',registration.raw({'cell':v.CELL,'decision_sha256':registration.sha(decision),
        'model_contacts':0,'cross_fold_output_equality_required':False}))
    try:
        result=repin_package(repo_root=repo,data_root=data,runtime_root=data,lock_path=lock,
            experiments=[(v.CELL,v.CELL)],readiness_dir=Path(registration.PKG),
            preregistration_path=Path(registration.PKG+'-preregistration.md'),
            decision=(v.DECISION_PATH,v.DECISION),set_ready=True,allow_output_change=False,run_check_fn=None)
        base=registration.read(repo/f'configs/baseline/{v.CELL}.json')
        pin=base['prepared_phase_supply'];path=v.comparator.inputs.confined(data,pin['relpath'])
        if not path.is_relative_to((data/'prepared'/v.RUN).resolve()) or registration.sha(path)!=pin['sha256']:
            raise ValueError('own Fold A supply pin differs')
        refreshed=refresh_package(repo,repo/registration.PKG)
        dependent=rewrite_dependent_hashes(repo_root=repo,readiness_dir=repo/registration.PKG,
            preregistration_path=repo/(registration.PKG+'-preregistration.md'),exposure_ledger_path=repo/'docs/exposure-ledger.md',
            run_ids={v.RUN,v.FIT},mapping={})
        if decision.read_bytes()!=before or result['checks']:raise ValueError('decision mutation or premature check')
        record={'repin':result,'refresh':refreshed,'dependent':dependent,'supply':pin,'model_contacts':0,
            'launchable':False,'remaining':['commit prepared configuration','actual preflight','guarded dispatch']}
        publish_new(out/'complete.json',registration.raw(record));return record
    except BaseException as error:
        publish_new(out/'failed.json',registration.raw({'reason':str(error),'automatic_retry':False}));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo-root','data-root','lock-path','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.repo_root,a.data_root,a.lock_path,a.out),sort_keys=True))


if __name__=='__main__':main()
