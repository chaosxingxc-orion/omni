"""Exercise dependencies, global concurrency and deterministic prompt order."""
import threading
import time
import pytest
from test_tfgrpo_learning import Model,group,learn


def test_ready_group_advances_before_unrelated_summary_finishes(tmp_path):
    blocked=threading.Event();advanced=threading.Event();model=Model()
    def caller(request):
        if request['stage']=='summary' and request['identity']=='q2-t0':
            blocked.set()
            assert advanced.wait(10),'global summary barrier remained'
        if request['stage']=='advantage' and request['identity']=='q1':
            assert blocked.wait(10)
            advanced.set()
        return model(request)
    _,result=learn(tmp_path,caller,[group(qid='q1'),group(qid='q2')],concurrency=4)
    assert advanced.is_set() and result['learning_usage']['calls']==9


@pytest.mark.parametrize('concurrency',[1,2,4])
def test_global_contact_bound_and_identical_prompts(tmp_path,concurrency):
    lock=threading.Lock();active=0;peak=0;model=Model()
    def caller(request):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:
            time.sleep(.015 if request['identity'].startswith('q1') else .001)
            with lock:return model(request)
        finally:
            with lock:active-=1
    groups=[group(qid='q1'),group(qid='q2')]
    _,result=learn(tmp_path/'parallel',caller,groups,concurrency=concurrency)
    serial=Model();learn(tmp_path/'serial',serial,groups,concurrency=1)
    key=lambda r:(r['stage'],r['identity'])
    assert {key(r):r['messages'] for r in model.requests}=={key(r):r['messages'] for r in serial.requests}
    assert peak<=concurrency and result['learning_usage']['calls']==9
    assert len(model.requests)==9
