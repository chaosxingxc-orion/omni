"""Check the original launcher's first resident continuation against real archived host bytes."""
import argparse,copy,io,json,sys,tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for k in ('run','archive','release','source','locked','precision','policy'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);a=p.parse_args()
    from stage_bounded_actor_step import stage_continuation
    sys.path.insert(0,str(a.release/'tree/parents/jev-system-rl/implementation'))
    from run_package import read,pin
    # Build from the same exact checked source that will be staged.
    data=a.archive.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        old=json.load(tar.extractfile('TRAIN.local.json'));commit=json.load(tar.extractfile(f'hot/step-{a.step}/COMMITTED.json'))
    assert old['devices']==list(range(8)) and commit['step']==a.step
    original=read(next((a.run/f'step-{a.step+1:06d}/update').glob('TRAIN-REQUEST*.json')))
    generation=copy.deepcopy(original['generation']);rt=generation['runtime']
    rt['reference_batch']=read(a.policy)['reference_batch'];rt['actor_fields']=read(a.policy)['actor_fields']
    name='fresh-cohort-successor/reference_batch.py';rt['sources']['reference_batch_source']=name
    mounts=copy.deepcopy(original['capture_mounts']);parents=a.release/'tree/parents'
    for path in [*mounts['overlays'],name]:mounts['overlays'][path]=pin(parents/path)['sha256']
    c={k:v for k,v in original.items() if k not in ('batch_selection_ref','measured_driver_ref')}
    c.update(run='stage-cpu-only',generation=generation,capture_mounts=mounts,check_only=True,stage_only=True,
        continuous_actor=dict(protocol='formal-resident-v1',source_lock_ref=pin(a.locked),precision_lock_sha256=pin(a.precision)['sha256']))
    path=Path(c['restore_run']);calls=[]
    def remote(cmd,*,text=True,**kw):
        import shlex
        expected=['tar','-cf','-','-C',str(path),'--','TRAIN.local.json','hot/EXIT.json',f'hot/step-{a.step}/COMMITTED.json',
            *(f'hot/rank-{i}/RESULT.json' for i in range(8))]
        if shlex.split(cmd)!=expected or text or kw:raise AssertionError('unexpected host action')
        calls.append(cmd);return data
    stage_continuation(c,a.source,path.with_name(c['run']),remote,timings={})
    assert len(calls)==1
    print(json.dumps(dict(stage='strict resident check_only',source_step=a.step,real_host_archive=True,device_actions=0)))
if __name__=='__main__':main()
