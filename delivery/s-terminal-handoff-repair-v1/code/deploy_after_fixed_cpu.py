"""Main-only local deployment after fixed suite. Never uses SSH/API/NPU/Git.

Does not enable policy or restart scheduler. Copies immutable package exclusively;
mutable host driver changes atomically only after package/native/helper are ready.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[3]
OWNER=ROOT/'delivery/s-terminal-handoff-repair-v1'
sys.path.insert(0,str(OWNER/'code'))
import terminal_handoff as h
from transport_manifest import verify
REQUIRED={'train_runtime.py','bind_train.py','train-runtime-v1/run_s_native.py','train-runtime-v1/native_resources.py'}

def exclusive_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        h.need(src.read_bytes()==dst.read_bytes(),'immutable destination differs: '+str(dst));return
    h.write_new(dst,src.read_bytes());h.syncdir(dst.parent)

def main():
    p=argparse.ArgumentParser();p.add_argument('--host-root',required=True)
    p.add_argument('--verified-attempt',required=True);a=p.parse_args()
    H=Path(a.host_root).resolve();attempt=Path(a.verified_attempt).resolve()
    h.need(H.is_dir() and attempt.is_relative_to(OWNER/'evidence'),'explicit existing H and owner attempt')
    receipt=attempt/'HANDOFF-TRAIN-VERIFIED.json';r=h.read(receipt)
    h.need(r['status']=='PASS_FIXED_IMAGE_REAL_TRAIN_HANDOFF_CPU' and r['formal_fixed_image'] is True
           and r['python']=='/opt/saea-venv/bin/python' and set(r['pins'])==REQUIRED,'real fixed suite receipt required')
    for item in [r['transport_manifest'],r['real_validation'],r['dependency_receipt']]:
        h.need(h.pin(item['path'])==item,'fixed evidence pin changed')
    verify(h.read(OWNER/'TRANSPORT-MANIFEST.json'))
    src=OWNER/'handoff-execution-code'
    for rel,pin in r['pins'].items():h.need(h.digest((src/rel).read_bytes())==pin,'new package pin changed')
    h.need(h.digest((src/'resources/tiktoken-cache'/r['resources_cache']['cache_key']).read_bytes())==r['resources_cache']['cache_sha256'],'resource cache pin')
    # Never touches H/execution-code or TRAIN96-VERIFIED; those may be mounted.
    for path in sorted(src.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:exclusive_copy(path,H/'handoff-execution-code'/path.relative_to(src))
    for rel,dst in [('code/terminal_handoff.py','terminal_handoff.py'),
                    ('code/native-runtime/run_s_native.py','handoff-native.py')]:
        exclusive_copy(OWNER/rel,H/dst)
    exclusive_copy(receipt,H/'HANDOFF-TRAIN-VERIFIED.json')
    policy=H/'terminal-handoff-policy.json'
    if not policy.exists():exclusive_copy(OWNER/'config/terminal-handoff-policy.default.json',policy)
    else:h.need(h.read(policy).get('enabled') is False,'deploy while policy disabled')
    driver=H/'host_driver.py'
    if driver.exists() and driver.read_bytes()!=(OWNER/'code/host_driver.py').read_bytes():
        exclusive_copy(driver,H/'code-history'/('host_driver.'+h.digest(driver.read_bytes())+'.py'))
    tmp=H/('host_driver.py.new-'+str(os.getpid()))
    h.write_new(tmp,(OWNER/'code/host_driver.py').read_bytes());os.replace(tmp,driver);h.syncdir(H)
    print(json.dumps(dict(status='FILES_DEPLOYED_POLICY_DISABLED',host_root=str(H),
          original_execution_code_untouched=True,scheduler_handover_still_required=True)))
if __name__=='__main__':main()
