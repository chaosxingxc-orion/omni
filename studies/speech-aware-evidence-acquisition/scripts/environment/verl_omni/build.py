"""Build an offline environment image only. Does not launch training or claim devices."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from prepare import verify, NGA_IMAGE

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context',type=Path,required=True)
    p.add_argument('--base-image',required=True,help='Local immutable sha256 image ID, never a floating tag')
    p.add_argument('--fallback',action='store_true',help='Rebuild from locked source on a base stack; new image identity')
    p.add_argument('--tag',default='omni-verl-lora-env:20261101')
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',a.base_image):p.error('immutable local image ID required')
    if a.receipt.exists() or a.receipt.resolve().is_relative_to(a.context.resolve()):p.error('fresh receipt outside build context required')
    verify(a.context)
    if not a.fallback and a.base_image != NGA_IMAGE:p.error('preferred path requires recorded NGA image ID')
    info=json.loads(subprocess.check_output(['docker','image','inspect',a.base_image]))[0]
    if info['Id']!=a.base_image or info['Architecture']!='arm64':raise ValueError('wrong local base identity or architecture')
    local_tag='omni-verl-offline-base:'+a.base_image.split(':')[1]
    subprocess.run(['docker','tag',a.base_image,local_tag],check=True)
    cmd=['docker','build','--pull=false','--network=none','--build-arg','BASE_IMAGE='+local_tag,
         '-f',str(a.context/('Dockerfile.offline' if a.fallback else 'Dockerfile')),'-t',a.tag,str(a.context)]
    subprocess.run(cmd,check=True,env={**os.environ,'DOCKER_BUILDKIT':'0'})
    image_id=subprocess.check_output(['docker','image','inspect',a.tag,'--format','{{.Id}}'],text=True).strip()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x') as f:
        json.dump({'image_id':image_id,'base_id':a.base_image,'time':time.time(),'command':cmd,
                   'builder':'legacy-local-no-pull','local_base_tag':local_tag,'context_verified':True,'fallback':a.fallback,'training_verified':False},f,indent=2)
    print(image_id)
