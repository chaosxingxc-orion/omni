"""Release a prepared comparison only if its measured parent earns the configured lock."""
import json
from pathlib import Path
import sys
import time

from io_utils import doc, pin, require, save
from lock_prefill_comparison import lock
from run_isolated_candidate import run


def advance(c):
    config=doc(pin(c['lock_config']))
    predecessor=Path(config['parent_packet'])
    result=Path(c['previous_score'])
    require(result==predecessor.with_name(predecessor.name+'-score'),'mismatched predecessor')
    deadline=time.monotonic()+c['wait_seconds']
    while not (result/'DRAINED.json').exists():
        require(time.monotonic()<deadline,'score deadline expired')
        time.sleep(5)
    require(doc(pin(result/'DRAINED.json'))['guards_releasable'] and doc(pin(result/'RESULT.json'))['rows'],
            'score ledger not sealed')
    evidence=lock(config)
    next_flow=doc(pin(c['next_flow']))
    cut=doc(pin(next_flow['cutover']))
    lane,=cut['lanes']
    declaration=Path(c['next_packet'])
    require(declaration.parent==Path(lane['new_code']).parent and cut['terminal']==str(result/'DRAINED.json')
            and doc(pin(declaration))['parent_sha256']==evidence['runtime_sha256'],'wrong successor')
    save(Path(c['lock_output']),evidence)
    run(next_flow)


if __name__=='__main__':
    advance(json.loads(Path(sys.argv[1]).read_bytes()))
