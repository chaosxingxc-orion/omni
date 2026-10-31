"""Wait for verified real500 completion, select, and hand off once to A0.

No partial-result selection, periodic model calls or automatic failed-job replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D,U,formal,write,status

RESULTS=[D/'work/codex/real500-tp4-s96-verified-result-v1.json',
    D/'work/codex/real500-result-queue-v1/slurp-real500-tp4-s128-v1.json',
    D/'work/codex/real500-result-queue-v1/slurp-real500-tp8-s256-v1.json']


def run(tree,out):
    if not out.resolve().is_relative_to(D/'work/codex') or out.exists():
        raise ValueError('fresh owned queue required')
    out.mkdir(parents=True)
    drivers=[U/'scripts/npu'/name for name in ('a0_selected_handoff.py','real500_successor_queue.py')]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pins={str(p):sha(p) for p in drivers}
    study=tree/'studies/speech-aware-evidence-acquisition'
    git=['git','-c','safe.directory='+str(study),'-C',str(study)]
    head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(git+['status','--porcelain']).strip():raise ValueError('clean continuation source required')
    write(out/'intent.json',{'pid':os.getpid(),'tree':str(tree),'source_commit':head,'driver_pins':pins,
        'interval_seconds':600,'automatic_retry':False,'authority':'existing owner real500 comparison and full A0 continuation'})
    status(out,'waiting_verified_full_comparison')
    try:
        while not all(p.exists() for p in RESULTS):
            terminal=D/'work/codex/real500-successor-queue-v1/slurp-real500-tp8-s256-v1/run.exit'
            if terminal.exists() and terminal.read_text().strip()!='0':raise ValueError('TP8 failed; preserve complete checkpoints')
            export_status=D/'work/codex/real500-result-queue-v1/status.json'
            if export_status.exists() and json.loads(export_status.read_bytes()).get('state')=='failed':
                raise ValueError('verified result export failed')
            time.sleep(600)
        if subprocess.check_output(git+['status','--porcelain']).strip() or subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()!=head:
            raise ValueError('continuation source changed while queued')
        if any(sha(Path(path))!=digest for path,digest in pins.items()):raise ValueError('handoff driver changed')
        status(out,'selecting_complete_results')
        selection=out/'selection.json'
        args=['scripts/analysis/select_real500_profile.py','--data',str(D),'--out',str(selection)]
        for p in RESULTS:args+=['--result-path',str(p)]
        formal(tree,args,out/'selection.log')
        decision=json.loads(selection.read_bytes())
        sources={'SAEA-SLURP-REAL500-TP4-S96-V1':D/'work/dev/real500-runtime',
            'SAEA-SLURP-REAL500-TP4-S128-V1':D/'work/dev/real500-successors',
            'SAEA-SLURP-REAL500-TP8-S256-V1':D/'work/dev/real500-successors-tp8'}
        status(out,'handoff',selected_profile=decision['profile'])
        with (out/'handoff.log').open('x') as log:
            subprocess.run(['python3',str(drivers[0]),'--tree',str(tree),
                '--source-tree',str(sources[decision['selected_run_id']]),'--selection',str(selection),
                '--out',str(out/'handoff')],stdout=log,stderr=subprocess.STDOUT,check=True)
        status(out,'a0_remaining_complete_full1080_score_pending')
    except BaseException as error:
        status(out,'failed',error=str(error),automatic_retry=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(**vars(p.parse_args()))
