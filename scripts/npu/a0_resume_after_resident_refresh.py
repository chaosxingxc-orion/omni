"""Resume one proven pre-dispatch A0 failure without switching services again."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D,U,IMAGE,formal,call,git,write,status
NAME='slurp-a0-remaining353-c2-v1'
PREVIOUS='slurp-real500-tp8-s256-v1'


def _resume(tree, failed, out, registered_stage=False):
    if not out.resolve().is_relative_to(D/'work/codex'): raise ValueError('owned recovery required')
    study=tree/'studies/speech-aware-evidence-acquisition'
    runtime=failed/'runtime';transition=D/'work/codex/frozen-engine-deployment/handoff'
    if registered_stage:
        state=json.loads((out/'status.json').read_bytes())
        if state['state']!='failed' or 'register recovered A0 resident' not in state.get('error',''):
            raise ValueError('expected empty-commit failure required')
        if git(study,'status','--porcelain'):raise ValueError('clean repinned source required')
        receipt=json.loads((runtime/'registration-complete.json').read_bytes())
        spec=json.loads((study/'docs/readiness'/NAME/'candidate.json').read_bytes())
        refresh=json.loads((study/'docs/readiness'/NAME/'resident-refresh.json').read_bytes())
        if receipt['session']!=spec['service_session'] or receipt['model_calls']!=0 or refresh['readiness']!='held':
            raise ValueError('completed held resident registration required')
        for path in (out/'run',out/'admit.log',D/'runs/SAEA-SLURP-A0-REMAINING353-C2-V1',D/'attempts/SAEA-SLURP-A0-REMAINING353-C2-V1',D/spec['decision_relpath']):
            if path.exists():raise ValueError('admission or inference already attempted')
        write(out/'empty-commit-failure.json',state)
        write(out/'registered-stage-resume.json',{'pid':os.getpid(),'time':time.time(),'new_model_calls':0})
    else:
        if out.exists() or not out.resolve().is_relative_to(D/'work/codex'):
            raise ValueError('fresh owned recovery required')
        old=json.loads((failed/'status.json').read_bytes())
        if old['state']!='failed' or 'refresh_a0_resident_binding.py' not in old.get('error',''):
            raise ValueError('expected pre-dispatch refresh failure required')
        for path in (failed/'run',failed/'run.exit.json',D/'runs/SAEA-SLURP-A0-REMAINING353-C2-V1',D/'attempts/SAEA-SLURP-A0-REMAINING353-C2-V1'):
            if path.exists():raise ValueError('possible existing generation; reconcile before recovery')
        if git(study,'status','--porcelain'):raise ValueError('committed partial held state required')
        runtime=failed/'runtime';transition=D/'work/codex/frozen-engine-deployment/handoff'
        out.mkdir(parents=True)
        write(out/'intent.json',{'pid':os.getpid(),'failed':str(failed),'source_commit':git(study,'rev-parse','HEAD'),
            'automatic_retry':False,'deployment_repeated':False,'time':time.time()})
        status(out,'finishing_recorded_resident_refresh')
        formal(tree,['scripts/analysis/resume_a0_resident_repin.py','--repo',str(study),'--data',str(D),
            '--runtime',str(runtime),'--transition',str(transition)],out/'repin.log')
        git(study,'diff','--check');git(study,'add','--','configs','docs/readiness','docs/receipts','docs/exposure-ledger.md')
        git(study,'commit','-m','fix: finish verified A0 resident repin without inference replay')
        formal(tree,['scripts/analysis/register_real_engine_resident.py','--repo',str(study),'--data',str(D),
            '--candidate',NAME,'--previous-candidate',PREVIOUS,'--runtime',str(runtime),
            '--transition',str(transition),'--archived-inspect',str(runtime/'archived-inspect.json')],out/'resident.log')
        git(study,'diff','--check');git(study,'add','--','configs','docs/readiness','docs/receipts','docs/exposure-ledger.md')
        if git(study,'diff','--cached','--name-only'):
            git(study,'commit','-m','chore: register recovered A0 resident')
        spec=json.loads((study/'docs/readiness'/NAME/'candidate.json').read_bytes())
    try:
        status(out,'admitting')
        formal(tree,['scripts/analysis/admit_real_engine_candidate.py','--repo',str(study),'--data',str(D),
            '--candidate',NAME],out/'admit.log')
        git(study,'diff','--check');git(study,'add','--','configs','docs/readiness','docs/exposure-ledger.md')
        git(study,'commit','-m','chore: admit exact missing A0 continuation')
        formal(tree,['scripts/checks/flight_registration_check.py','--repo-root',str(study),'--data-root',str(D),
            '--cell',NAME,'--json'],out/'registration-check.log')
        formal(tree,['-m','speech_aware_evidence_acquisition.core.flight.cli','check','--repo-root',str(study),
            '--data-root',str(D),'--runtime-root',str(D/'work/scratch/runtime-root'),'--lock-path',str(tree/'docs/datasets.lock.json'),
            '--experiment',NAME,'--cell',NAME,'--read-only'],out/'preflight.log',spec['profile'])
        status(out,'running353',source_commit=git(study,'rev-parse','HEAD'))
        env={**os.environ,'TREE':str(tree),'CPUSET':'32-63','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,
            'SAEA_DEPLOYMENT_PROFILE':spec['profile'],
            'PORT_DOCKER_FLAGS':'--pid host --network host --name saea-a0-remaining353-generation -e SAEA_STAGE_TIMING_PATH='+str(out/'stages.jsonl')}
        call(['bash',str(D/'work/scripts/port_py.sh'),'scripts/analysis/run_real_engine_candidate.py',
            '--repo',str(study),'--data',str(D),'--candidate',NAME,'--out',str(out/'run')],out/'run.log',env)
        write(out/'run.exit.json',{'exit':0,'time':time.time()})
        status(out,'complete353_full1080_score_pending')
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_retry=False)
        raise


def resume(tree, failed, out, registered_stage=False):
    try:
        return _resume(tree, failed, out, registered_stage)
    except BaseException as error:
        if out.exists() and out.resolve().is_relative_to(D/'work/codex'):
            status(out,'failed',error=str(error),automatic_retry=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('tree','failed','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--registered-stage',action='store_true')
    resume(**vars(p.parse_args()))
