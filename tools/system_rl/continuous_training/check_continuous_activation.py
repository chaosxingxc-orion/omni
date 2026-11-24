"""Exercise the actual ASGI admission lock around an in-flight old-generation call."""
import argparse,asyncio,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

async def run(mod,root):
    old,new='old','new'
    for version,i in [(old,1),(new,2)]:
        (root/version).write_text(json.dumps({'step':i}))
    oldref,newref=(mod.pin(root/v) for v in (old,new))
    started=asyncio.Event();finish=asyncio.Event()
    class Overlay:
        def __init__(self,*a,**kw):
            self.service_identity=kw['service_identity'];self.switches=0
            self.engine=SimpleNamespace(reset_prefix_cache=self.reset)
        async def reset(self,**kwargs):return True
        async def activate(self,ref,profile):
            self.active={'policy_ref':ref,'policy_version':Path(ref['path']).name,'adapter_id':1+self.switches,
                         'service_identity':self.service_identity}
            self.activation_ref=ref
        async def replace_checkpoint(self,ref,*args):
            assert not started.is_set() or finish.is_set()
            self.switches+=1
            await self.activate(ref,None)
        async def __call__(self,*args):
            version=self.active['policy_version'];started.set()
            await finish.wait();assert self.active['policy_version']==version
            return version
    async def unexpected(*a): raise AssertionError('unexpected business/service invocation')
    async def install(*a): pass
    cfg={'overlay_root':str(root),'endpoint':'test','service_identity':{'pid':1},'policy_ref':oldref,
         'profile_ref':{},'named_rpc_overlay':{'precision_ref':{}}}
    mw=mod.policy_middleware(cfg,{'placeholder':1},Overlay,{'model':'test','bounded_candidate':True},str(root),
        'formal',True,lambda _:True,install)(unexpected)
    async def call(path,peer='127.0.0.1'):
        msgs=[]
        async def send(x): msgs.append(x)
        result=await mw({'type':'http','path':path,'client':(peer,1),'method':'POST'},unexpected,send)
        return result if not msgs else (msgs[0]['status'],json.loads(msgs[-1]['body'] or b'null'))
    control='/_omni/activation'
    assert (await call(control,'192.0.2.1'))[0]==403 and mw.overlay is None
    assert (await call(control))[1]['version']==old and not started.is_set()
    first=asyncio.create_task(call('/v1/chat/completions'));await started.wait()
    (root/'after-step-1.json').write_text('{}')
    with patch.object(mod,'local_successor',return_value=newref):
        change=asyncio.create_task(call(control));await asyncio.sleep(0)
        assert not change.done() and mw.overlay.switches==0
        finish.set();assert await first==old
        assert (await change)[1]['version']==new
        assert (await call(control))[1]['version']==new and mw.overlay.switches==1
        assert await call('/v1/chat/completions')==new
    print(json.dumps(dict(test='ASGI admission with stub engine',old_call_finished=True,
        switched_once=True,control_generated_tokens=0,retry_returns_same_version=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(Path(__file__).parent/'source'),str(a.release/'tree/parents/policy-iteration-successor')]
    import service_bootstrap
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp: asyncio.run(run(service_bootstrap,Path(tmp)))
