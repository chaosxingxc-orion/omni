"""Deterministically exercise producer/consumer overlap and abort before the step."""
from threading import Event
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from microbatch_fields import overlap_microbatches


def check(fail):
    preparing_second=Event(); consuming_first=Event(); events=[]; updated=False
    def prepare(index):
        if index:
            preparing_second.set()
            if not consuming_first.wait(5):raise AssertionError('producer and consumer were serial')
            if fail:raise ValueError('second wave failed')
        return index
    try:
        for index in overlap_microbatches(range(2),prepare):
            if index==0:
                if not preparing_second.wait(5):raise AssertionError('second wave not started')
                consuming_first.set()
            events.append(index)
        updated=True
    except ValueError as exc:
        if not fail or str(exc)!='second wave failed':raise
    if events!=([0] if fail else [0,1]) or updated==fail:raise ValueError('invalid batch boundary')
    return dict(consumed=events,step_reached=updated)

if __name__=='__main__':print(json.dumps(dict(success=check(False),producer_failure=check(True))))
