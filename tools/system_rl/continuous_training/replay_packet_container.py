"""Replay a completed training container's read-only inputs on CPU in its exact image."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
def run(a):
    info=json.loads(subprocess.check_output(['docker','inspect',a.container]))[0]
    if info['State']['Status']!='exited' or info['State']['ExitCode']!=0:raise ValueError('expected completed training container')
    if subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(a.scratch)],text=True).strip() not in ('ext4','xfs','btrfs'):raise ValueError('need local storage')
    fs=os.statvfs(a.scratch)
    if fs.f_bavail*fs.f_frsize<24*2**30 or fs.f_bavail/fs.f_blocks<.1 or fs.f_favail/fs.f_files<.05:raise ValueError('scratch capacity')
    out=Path(tempfile.mkdtemp(prefix='packet-replay-',dir=a.scratch))
    env=dict(TMPDIR='/scratch/tmp',XDG_CACHE_HOME='/scratch/cache',HF_HOME='/scratch/hf',TORCHINDUCTOR_CACHE_DIR='/scratch/inductor',TRITON_CACHE_DIR='/scratch/triton',HOME='/scratch/home',ASCEND_WORK_PATH='/scratch/work',ASCEND_CACHE_PATH='/scratch/ascend-cache',ASCEND_PROCESS_LOG_PATH='/scratch/home/ascend/log')
    for v in env.values():(out/Path(v).relative_to('/scratch')).mkdir(parents=True,exist_ok=True)
    cmd=['docker','run','--rm','--network=none','--read-only','--user',info['Config']['User'],'--memory',a.memory,'--cpus',str(a.cpus),'--entrypoint','python3','-w','/scratch']
    for m in info['Mounts']:
        if m['Type']!='bind':raise ValueError('unexpected original mount type')
        cmd.extend(['-v',m['Source']+':'+m['Destination']+':ro'])
    cmd.extend(['-v',str(out)+':/scratch:rw','-v',str(Path(__file__).resolve().parent)+':/new-sources:ro'])
    for item in info['Config']['Env']:
        k,v=item.split('=',1)
        if k in ('PYTHONPATH','PATH','LD_LIBRARY_PATH') or k.startswith('OMNI_'):env[k]=v
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for k,v in env.items():cmd.extend(['-e',k+'='+v])
    print(out,flush=True)
    subprocess.run(cmd+[info['Image'],'-B','/new-sources/check_accepted_actor_packet.py','--spec',a.spec,'--root',a.root,'--out','/scratch','--ranks',str(a.ranks)],check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('container','scratch','spec','root','memory'):p.add_argument('--'+k,required=True)
    for k in ('ranks','cpus'):p.add_argument('--'+k,type=int,required=True)
    run(p.parse_args())
