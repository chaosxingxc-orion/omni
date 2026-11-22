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
    terminal=Path(cut['terminal'])
    if 'after_score' in cut:
        source=Path(cut['after_score'])
        require(terminal==source.parent/'DRAINED.json','must wait for own predecessor')
        until=time.monotonic()+cut['wait_seconds']
        while not(source.is_file() and terminal.is_file()):
            require(time.monotonic()<until,'predecessor not sealed');time.sleep(3)
        require(doc(pin(terminal))['guards_releasable'] and doc(pin(source))['rows'],'invalid predecessor')
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

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    run(doc(pin(p.parse_args().config)))
