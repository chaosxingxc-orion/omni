"""Execute one reviewed selected-profile handoff to the exact missing A0 rows.

Selection must already pin all complete real500 results. Every operation is
create-only; an interrupted handoff is reconciled from evidence, never replayed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from real500_successor_queue import D,U,IMAGE,formal,call,git,capture,write,status

NAME='slurp-a0-remaining353-c2-v1'
PREVIOUS='slurp-real500-tp8-s256-v1'


def handoff(tree,source_tree,selection,out):
    if not out.resolve().is_relative_to(D/'work/codex') or out.exists():
        raise ValueError('fresh owned handoff output required')
    study=tree/'studies/speech-aware-evidence-acquisition'
    source=source_tree/'studies/speech-aware-evidence-acquisition'
    if git(study,'status','--porcelain') or git(source,'status','--porcelain'):
        raise ValueError('clean source trees required')
    terminal=D/'work/codex/real500-successor-queue-v1'/PREVIOUS/'run.exit'
    if not terminal.exists() or terminal.read_text().strip()!='0':
        raise ValueError('complete successful TP8 comparison required')
    decision=json.loads(selection.read_bytes())
    if decision['selected_run_id'] not in ('SAEA-SLURP-REAL500-TP4-S96-V1','SAEA-SLURP-REAL500-TP4-S128-V1','SAEA-SLURP-REAL500-TP8-S256-V1'):
        raise ValueError('unsupported selected topology; review another handoff')
    out.mkdir(parents=True)
    try:
        write(out/'intent.json',{'selection':str(selection),'source_tree':str(source_tree),'tree':str(tree),
            'source_commit':git(source,'rev-parse','HEAD'),'initial_commit':git(study,'rev-parse','HEAD'),
            'candidate':NAME,'time':time.time(),'automatic_retry':False})
        status(out,'binding_complete_comparison')
        formal(tree,['scripts/analysis/bind_a0_selected_profile.py','--repo',str(study),'--data',str(D),
            '--source-repo',str(source),'--selection-path',str(selection)],out/'bind.log')
        git(study,'diff','--check')
        git(study,'add','--','configs','docs/readiness','docs/receipts','docs/exposure-ledger.md')
        git(study,'commit','-m','chore: bind missing A0 rows to measured engineering profile')
        spec=json.loads((study/'docs/readiness'/NAME/'candidate.json').read_bytes())
        previous=json.loads((study/'docs/readiness'/PREVIOUS/'candidate.json').read_bytes())
        if spec['profile']!=previous['profile']:
            status(out,'switching_to_selected_profile')
            metrics=U/'scripts/npu/real_engine_metrics.py'
            call(['python3',str(metrics),'--port','8200','--output',str(out/'tp8-final-metrics.json')],out/'metrics.log')
            transition=D/'work/codex/frozen-engine-deployment'/out.name
            switch=U/'scripts/npu/freeze_engine/topology_switch.py'
            call(['python3',str(switch),'plan','--source',previous['profile'],'--target',spec['profile'],'--record',str(transition)],out/'plan.log')
            call(['python3',str(switch),'apply','--record',str(transition)],out/'apply.log')
            runtime=out/'runtime';capture(study,transition,runtime)
            formal(tree,['scripts/analysis/refresh_a0_resident_binding.py','--repo',str(study),'--data',str(D),
                '--runtime',str(runtime),'--transition',str(transition)],out/'refresh.log')
            formal(tree,['scripts/analysis/register_real_engine_resident.py','--repo',str(study),'--data',str(D),
                '--candidate',NAME,'--previous-candidate',PREVIOUS,'--runtime',str(runtime),
                '--transition',str(transition),'--archived-inspect',str(runtime/'archived-inspect.json')],out/'resident.log')
            git(study,'diff','--check');git(study,'add','--','configs','docs/readiness','docs/receipts','docs/exposure-ledger.md')
            git(study,'commit','-m','chore: bind fresh selected-profile resident for A0 continuation')
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('tree','source-tree','selection','out'):p.add_argument('--'+key,type=Path,required=True)
    handoff(**vars(p.parse_args()))
