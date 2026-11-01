"""Print or launch an offline CPU/meta probe on the recovered npu2 host, never training."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--name',default='omni-verl-env-probe-v1')
    p.add_argument('--cpuset',default='208-215');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',a.image):p.error('immutable image ID required')
    if not re.fullmatch(r'omni-verl-[a-z0-9-]+',a.name):p.error('owned container name required')
    if not re.fullmatch(r'[0-9,-]+',a.cpuset):p.error('explicit CPU set required')
    root=Path('/mnt/cephfs/xingchao/omni').resolve();model=a.model.resolve();out=a.out.resolve()
    if not model.is_relative_to(root/'models') or not out.is_relative_to(root/'work'):p.error('owned model/output paths required')
    if out.exists():p.error('fresh output directory required; inspect old terminal before retrying')
    # No davinci compute devices. Management/driver libraries may be needed for torch_npu import.
    cmd=['docker','run','-d','--name',a.name,'--network=none','--cpuset-cpus',a.cpuset,'--shm-size=8g',
         '--mount',f'type=bind,source={model},target=/model,readonly',
         '--mount',f'type=bind,source={out},target=/out',
         '--mount','type=bind,source=/usr/local/Ascend/driver,target=/usr/local/Ascend/driver,readonly',
         '--device=/dev/davinci_manager','--device=/dev/devmm_svm','--device=/dev/hisi_hdc',
         '-e','ASCEND_RT_VISIBLE_DEVICES=','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
         '--entrypoint','bash',a.image,'-c',
         'python3 -P /opt/omni-verl/probe.py --model /model --out /out/probe.json > /out/probe.log 2>&1; code=$?; printf "%s\\n" "$code" > /out/EXIT; exit "$code"']
    print(shlex.join(cmd))
    if a.execute:
        info=json.loads(subprocess.check_output(['docker','image','inspect',a.image]))[0]
        if info['Id']!=a.image or info['Architecture']!='arm64':raise ValueError('image identity or architecture mismatch')
        out.mkdir(parents=True,exist_ok=False)
        (out/'launch.json').write_text(json.dumps({'argv':cmd,'image':a.image,'model':str(model),'compute_devices':[],'training':False},indent=2)+'\n')
        subprocess.run(cmd,check=True)
