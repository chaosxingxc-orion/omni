"""Run the isolated fixed-action panel after a measured prefill promotion."""
import sys
from pathlib import Path
from cutover import switch
from io_utils import doc, pin, require, save
from lock_prefill_comparison import lock
from run_engine_comparison import main as measure
import subprocess
import time


def run(c):
    cut = doc(pin(c['cutover']))
    lane, = cut['lanes']
    lock_config = doc(pin(c['lock_config']))
    parent = Path(lock_config['parent_packet'])
    require(Path(cut['after_score']) == parent.with_name(parent.name+'-score')/'RESULT.json'
            and Path(cut['terminal']) == Path(cut['after_score']).with_name('DRAINED.json'),
            'comparison parent does not own preceding score')
    until = time.monotonic() + cut['wait_seconds']
    while not Path(cut['terminal']).is_file():
        require(time.monotonic()<until,'matched parent did not drain')
        time.sleep(5)
    require(doc(pin(cut['terminal']))['guards_releasable'], 'busy parent')
    evidence=lock(lock_config)
    root=Path(lane['new_code']).parent
    require(doc(pin(root/'REPLAY-DELTA.json'))['parent_sha256']==evidence['runtime_sha256'],
            'replay parent differs from measured candidate')
    save(c['lock_output'],evidence)
    require(not Path(cut['output']).exists(),'existing cutover requires reconciliation')
    subprocess.run([sys.executable,'-B',str(Path(lane['new_code'])/'launch.py'),
                    'check','--attempt',lane['preflight']],check=True)
    switch(cut)
    measure(doc(pin(c['measurement'])))


if __name__=='__main__':
    run(doc(pin(sys.argv[1])))
