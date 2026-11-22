"""Resolve the new container receipt once the specified latency window has ended."""
from pathlib import Path
import sys,time
from io_utils import doc,pin,require
from probe_native_scores import run


def main(c):
    path=Path(c['prerequisite']); cfg=c['config']
    require(path==Path(cfg['after']).with_name('DRAINED.json'),'unrelated prerequisite')
    end=time.monotonic()+c['wait_seconds']
    while not path.is_file():
        require(time.monotonic()<end,'latency not terminal');time.sleep(5)
    require(doc(pin(path))['guards_releasable'] and doc(pin(cfg['after']))['passed'],'latency failed')
    run(dict(cfg,start=pin(c['start_receipt'])))

if __name__=='__main__':main(doc(pin(sys.argv[1])))
