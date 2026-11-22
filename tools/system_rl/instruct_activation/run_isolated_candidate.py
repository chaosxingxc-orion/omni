"""Check, cut over and measure one candidate after its predecessor has drained."""
import argparse,time
from pathlib import Path
import subprocess,sys
from io_utils import doc,pin,require
from cutover import switch
from run_engine_comparison import main as measure
from probe_after_cutover import main as scores

def run(c):
    cut=doc(pin(c['cutover']))
    if 'after_no_promotion' in c:
        decision=Path(c['after_no_promotion'])
        lane,=cut['lanes']
        previous=Path(lane['old_code']).parent
        require(decision in (previous.with_name(previous.name+'-DECISION.json'),
                             previous.with_name(previous.name+'-native-DECISION.json')), 'decision from wrong study')
        until=time.monotonic()+cut['wait_seconds']
        while not decision.is_file():
            require(time.monotonic()<until,'predecessor decision missing')
            time.sleep(5)
        if doc(pin(decision))['action']!='no_promotion':
            return
    terminal=Path(cut['terminal'])
    if 'after_score' in cut:
        source=Path(cut['after_score'])
        require(terminal==source.parent/'DRAINED.json','must wait for own predecessor')
        until=time.monotonic()+cut['wait_seconds']
        while not(source.is_file() and terminal.is_file()):
            require(time.monotonic()<until,'predecessor not sealed');time.sleep(3)
        require(doc(pin(terminal))['guards_releasable'] and doc(pin(source))['rows'],'invalid predecessor')
    elif 'after_iterations' in cut:
        from cutover import load
        from io_utils import digest
        lane, = cut['lanes']
        old = load(lane['old_code'], 'diagnostic_owner')
        previous = Path(lane['old_code']).parent
        require(Path(cut['after_iterations']['path']).parent == previous, 'wrong diagnostic owner')
        cfg = doc(pin(terminal.parent.parent/'CONFIG.local.json'))
        result = doc(pin(terminal.parent/'RESULT.json'))
        rr, = result['rounds']
        arm, = [s for s in cfg['services'] if s['name'] == cut['diagnostic_lane']]
        state = old.inspect(old.name(lane['old_attempt']))
        proof = doc(cut['after_iterations'])
        require(arm['container'] == old.name(lane['old_attempt']) and arm['pid'] == state['State']['Pid']
                and state['State']['Running'] and state['Id'] == doc(cut['diagnostic_receipt'])['container_id']
                and Path(cfg['output']) == terminal.parent and result['passed']
                and doc(pin(terminal))['guards_releasable'] and not list(terminal.parent.rglob('UNKNOWN.json'))
                and rr['sha256'] == proof['round_sha256'] and proof['rows'], 'unrelated diagnostic result')
    elif 'after_forward' in cut:
        from cutover import load
        from decode_lineage import snapshot
        from io_utils import digest
        proof_path=Path(cut['after_forward'])
        until=time.monotonic()+cut['wait_seconds']
        while not proof_path.is_file():
            require(time.monotonic()<until,'parent training precision not sealed');time.sleep(5)
        proof=doc(pin(proof_path))
        lane,=cut['lanes']
        previous=Path(lane['old_code']).parent
        observed=doc(pin(previous/'CONFIG.local.json'))
        result=doc(pin(terminal.parent/'RESULT.json'))
        rr,=result['rounds']
        require(proof['passed'] and proof['optimizer_updates']==0 and proof['unresolved_top1']==0
                and proof['observer_round']==rr['sha256'] and result['passed']
                and doc(pin(terminal))['guards_releasable'] and not list(terminal.parent.rglob('UNKNOWN.json'))
                and Path(observed['output'])==terminal.parent==previous/'rounds','unrelated observer')
        require(proof['runtime']==digest(snapshot(Path(cut['parent_packet'])))==
                doc(cut['candidate_delta'])['immediate_parent'] and
                Path(cut['candidate_delta']['path']).parent==Path(lane['new_code']).parent,'different parent')
        old=load(lane['old_code'],'observer_owner'); ctr,=observed['services']
        state=old.inspect(old.name(lane['old_attempt']))
        require(ctr['container']==old.name(lane['old_attempt'])
                and state['Id']==doc(pin(observed['start_receipts'][ctr['name']]))['container_id']
                and state['State']['Running'],'observer changed')
    else:
        from reconcile_failed_panel import inspect_window
        cfg=Path(cut['failed_window'])
        previous=doc(pin(cut['failed_run']))
        require(terminal == Path(doc(pin(cfg))['output'])/'DRAINED.json','unrelated failed window')
        evidence=inspect_window(cfg,previous['start_receipts'])
        require(doc(pin(terminal))==evidence,'failed window no longer matches')
    require(not Path(cut['output']).exists(),'existing cutover requires reconciliation')
    for lane in cut['lanes']:
        code=Path(lane['new_code'])
        subprocess.run([sys.executable,'-B',str(code/'launch.py'),'check','--attempt',lane['preflight']],check=True)
    switch(cut)
    measure(doc(pin(c['measure'])))
    scores(doc(pin(c['scores'])))
    if 'parent_forward_report' in c:
        from report_prefill_candidate import report
        from io_utils import save
        spec=doc(c['parent_forward_report'])
        if 'after_forward' in cut:
            spec.update(training_proof=pin(cut['after_forward']))
        require(doc(spec['training_proof'])['natural_round']==
                doc(pin(Path(spec['parent_config']).parent/'rounds'/'RESULT.json'))['rounds'][0]['sha256'],
                'training reference does not match performance parent')
        save(spec['output'],report(spec))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    run(doc(pin(p.parse_args().config)))
