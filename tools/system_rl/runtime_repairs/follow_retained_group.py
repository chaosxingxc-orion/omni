"""Hand off one closed fresh TP4 result to the original guarded HF4 launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from dispatch_retained_hf_boundary import save


def run(cfg):
    src=Path(cfg['tp_output'])
    dst=Path(cfg['hf_bundle'])
    ticket=json.loads((src/'CONTAINER.json').read_bytes())
    cid=ticket['container_id']
    code=int(subprocess.check_output(['docker','wait',cid],text=True))
    if code:
        raise ValueError('TP4 exited unsuccessfully; refusing HF handoff')
    for _ in range(60):
        if (src/'EXIT.json').exists():break
        time.sleep(1)
    closed=json.loads((src/'EXIT.json').read_bytes())
    if closed['container_id']!=cid or closed['exit_code']!=0 or not closed['result_exists']:
        raise ValueError('TP4 closeout does not match live container')
    s=cfg['hf_spec']
    result=(src/'hot/RESULT.json').read_bytes()
    r=json.loads(result)
    if (r['fresh_group']['manifest_sha256']!=s['files']['FRESH.json']
            or r['actual_tp']!=4 or r['sampled_actions']!=sum(len(row['token_ids']) for row in r['fresh_group']['records'])):
        raise ValueError('TP4 did not use the expected source or generation scope')
    digest=hashlib.sha256(result).hexdigest()
    p=dst/s['fresh_group']['result']
    if p.exists():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('existing result differs')
    else:
        with p.open('xb') as f:
            f.write(result)
    s['files'][p.name]=digest
    target=dst/'HF-GROUP.local.json'
    if target.exists():raise ValueError('handoff already issued: inspect original launcher')
    save(target,s)
    subprocess.run([sys.executable,'-B',str(dst/'launch_retained_hf_fsdp.py'),'--config',str(target)],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    run(json.loads(p.parse_args().config.read_bytes()))
