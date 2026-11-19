"""Read only sealed local tensor pairs and run the canonical boundary comparison."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import os
import shlex
import subprocess
import time

from compare_retained_audio_boundary import run


def read_remote(host, *argv):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                                   host,shlex.join(argv)], timeout=90)


def store(path, data):
    if path.exists():
        if path.read_bytes()!=data: raise ValueError('conflicting closed readback')
        return
    part=path.with_suffix(path.suffix+'.pending')
    with part.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(part,path)
    fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def collect(side, config, out, layer=None):
    host=config['host'];root=Path(config['root']);lease=root/'hot/new-lease'
    deadline=time.monotonic()+config['timeout_seconds']
    # The two launchers run independently after their existing owners restore.
    while time.monotonic()<deadline:
        seen=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
                             shlex.join(['test','-f',str(lease/'NPU-EXIT.json')])], timeout=20)
        if seen.returncode==0:break
        if seen.returncode!=1:raise RuntimeError('remote observation unavailable')
        time.sleep(30)
    else:raise TimeoutError('no terminal phase receipt; preserve and reconcile original owner')
    get=lambda path:read_remote(host,'cat',str(path))
    state=json.loads(get(lease/'NPU-EXIT.json'))
    if state['Running'] or state['ExitCode']!=0:raise ValueError(f'{side} replay failed: '+str(state['ExitCode']))
    blob=get(root/'hot/RESULT.json');doc=json.loads(blob)
    if side=='hf':
        assert doc['inputs']['behavior']['sha256']==config['behavior_sha256']
        files={'HF.safetensors':doc['tensors_sha256']}
    else:
        spec=json.loads(get(root/'REPLAY.json'))
        assert spec['behavior']['sha256']==config['behavior_sha256']
        files={f"rank-{r['rank']}/VL.safetensors":r['sha256'] for r in doc['ranks']}
    if layer is not None:
        for rank in ([doc] if side=='hf' else doc['ranks']):
            prefix='' if side=='hf' else f"rank-{rank['rank']}/"
            meta=get(root/'hot'/(prefix+'LAYER.json'));audit=json.loads(meta)
            assert audit['layer']==layer and audit['sha256']==rank['layer_sha256']
            dst=out/side/(prefix+'LAYER.json');dst.parent.mkdir(parents=True,exist_ok=True);store(dst,meta)
            files[prefix+'LAYER.safetensors']=rank['layer_sha256']
    for name,digest in files.items():
        data=get(root/'hot'/name)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('closed operand digest differs')
        dst=out/side/name;dst.parent.mkdir(parents=True,exist_ok=True);store(dst,data)
    store(out/side/'RESULT.json',blob)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_bytes());out=Path(c['out']);out.mkdir(parents=True,exist_ok=True)
    assert c['hf']['behavior_sha256']==c['vl']['behavior_sha256']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(collect,k,c[k],out,c.get('inspect_layer')) for k in ('hf','vl')]
        for job in jobs:job.result()
    report=run(out);store(out/'COMPARE.json',json.dumps(report,indent=2).encode())
    if 'inspect_layer' in c:
        from compare_retained_layer import compare
        store(out/'LAYER-COMPARE.json',json.dumps(compare(out,c['inspect_layer']),indent=2).encode())
    print(json.dumps({r['rank']:r['first_difference'] for r in report['ranks']}))
