"""Offline dispatcher regression: failed unknown attempts must not stop a lane."""
import ast
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).with_name('scheduler.py')

def functions(names):
    tree=ast.parse(SOURCE.read_text())
    return compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(SOURCE),'exec')

def check_wait(unknown):
    terminal=dict(status='FAILED' if unknown else 'COMPLETE',unknown=unknown)
    def snapshot(lane,launch): return dict(terminal=terminal)
    ns=dict(HOSTS={'D':'test'},WINDOW={'end_epoch':time.time()+100},threading=threading,time=time,
            wake=threading.Event(),run=lambda *a,**kw:SimpleNamespace(stdout='1' if unknown else '0'),snapshot=snapshot)
    exec(functions({'observation_due','wait_container'}),ns)
    result,_=ns['wait_container']('D',dict(container_id='old',state='/unused'))
    assert result==terminal

def check_new_task_continues():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);events=[];chosen=iter([dict(task_id='failed'),dict(task_id='independent')])
        class Thread:
            def __init__(self,target,args,**kw):self.target,self.args=target,args
            def start(self):self.target(*self.args)
        window={'end_epoch':time.time()+100}
        initial=dict(predecessor=dict(container='known-good',terminal='/known/TERMINAL.json'),runtime={'unchanged':True})
        def read(path):
            if str(path).endswith('initial-lanes.json'):return {'D':initial}
            raise AssertionError('unexpected read '+str(path))
        def save(path,data):events.append(('save',str(path),data))
        def remote_json(host,args):
            tid='independent' if 'independent' in args[-1] else 'failed'
            return dict(task_id=tid,container_id=tid,state='/jobs/'+tid,runtime=initial['runtime'])
        def wait(lane,launch):
            if launch['task_id']=='failed':return dict(status='FAILED',unknown=1),dict(exit_code=1)
            window['end_epoch']=0
            return dict(status='COMPLETE',unknown=0),dict(exit_code=0)
        ns=dict(HOSTS={'D':'host'},ROOTS={'host':'/remote'},O=root,WINDOW=window,time=time,
                read=read,save=save,choose=lambda lane:next(chosen),stage=lambda task,host:'/workspace',
                launch_command=lambda *a:[],run=lambda *a,**kw:SimpleNamespace(stdout=''),remote_json=remote_json,
                wait_container=wait,update=lambda t,**kw:events.append(('update',t['task_id'],kw)),
                score=lambda *a:events.append(('score',a[1]['task_id'])),
                subprocess=SimpleNamespace(run=lambda *a,**kw:None),threading=SimpleNamespace(Thread=Thread),
                hashlib=__import__('hashlib'),traceback=__import__('traceback'),wake=threading.Event())
        exec(functions({'lane_worker'}),ns);ns['lane_worker']('D')
        updates=[e for e in events if e[0]=='update']
        assert [(e[1],e[2]['status']) for e in updates]==[('failed','running'),('failed','failed'),('independent','running'),('independent','completed')],events
        assert not any(e[0]=='save' and e[1].endswith('-BLOCKED.json') for e in events),events
        specs=[e[2] for e in events if e[0]=='save' and '/specs/' in e[1]]
        assert specs[1]['predecessor']==initial['predecessor'],'failed request reused as predecessor'

if __name__=='__main__':
    check_wait(0);check_wait(1);check_new_task_continues()
    print(json.dumps(dict(status='PASS',tests=['known_complete_returned','known_unknown_terminal_returned','unknown_preserved_independent_successor_started'],NPU_calls=0)))
