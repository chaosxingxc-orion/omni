"""Read-only closure of a rejected resident after independent restoration.

No dispatch, service stop, retry, cost refund or numerical acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import hashlib,json,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]);root=Path(c['remote_root']);control=root/'control/package/hot/new-lease'
def read(p):return json.loads(p.read_text())
def inspect(cid):
 v=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
 assert v['Id']==cid
 return v
chain=read(root/'CONTROLLER-RECOVERY-CHAIN.json')
assert chain[-1]['container_id']==c['controller']
controller=inspect(c['controller']);actor=inspect(c['actor'])
watch_id=read(control/'WATCHDOG.json')['container_id'];watch=inspect(watch_id)
for obj in [controller,actor,watch]:assert obj['State']['Status']=='exited' and not obj['State']['Running']
assert controller['State']['ExitCode']!=0 and actor['State']['ExitCode']!=0 and watch['State']['ExitCode']==0
assert read(control/'NPU-LAUNCH.json')['container_id']==actor['Id']
assert read(control/'NPU-EXIT.json')==actor['State']
restored=read(control/'RESTORED.json');assert restored['config_exact'] is True
plan=read(root/'control/package/LIFECYCLE-PLAN.json')
services=[]
for service in plan['services']:
 actual=inspect(service['id']);prior=read(Path(service['inspect_ref']['path']))
 assert actual['State']['Running']
 assert all(actual[k]==prior[k] for k in ('Id','Image','Config','HostConfig'))
 assert any(v['container_id']==actual['Id'] and v['config_exact'] for v in restored['services'])
 services.append(dict(id=actual['Id'],running=True,config_exact=True))
closed=root/'control/package/hot/resident/results/000000/RETURN.json';raw=closed.read_bytes()
assert hashlib.sha256(raw).hexdigest()==c['result_sha256']
result=json.loads(raw);assert result['passed'] is False and result['optimizer_steps']==0 and result['usable_for_update'] is False
assert result['charged_tokens_upper']==c['charged_tokens_upper']
print(json.dumps(dict(schema='failed-resident-terminal-proof-v1',source_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),controller=dict(id=controller['Id'],state=controller['State']),actor=dict(id=actor['Id'],state=actor['State']),watchdog=dict(id=watch_id,state=watch['State']),services=services,result_sha256=c['result_sha256'],charged_tokens_upper=c['charged_tokens_upper'],optimizer_steps=0,training_admission=False,restored=True,all_owners_terminal=True,model_calls=0)))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    assert cfg['schema']=='failed-resident-reconciliation-v1'
    cursor=json.loads(Path(cfg['cursor']).read_text())
    assert cursor['phase']=='done' and cursor['step']==0 and cursor['pending_stage'] is None
    outcome=json.loads((Path(cfg['cursor']).parent/'OUTCOME.json').read_text())
    assert outcome['status']=='DIAGNOSTIC_REJECTED'
    assert cursor['charged_tokens_upper']==cfg['remote']['charged_tokens_upper']
    result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',cfg['host'],
        shlex.join(['python3','-B','-c',REMOTE,json.dumps(cfg['remote'])])],timeout=60,text=True)
    proof=json.loads(result);proof['config_sha256']=hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    out=Path(cfg['output']);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        json.dump(proof,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    fd=os.open(out.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(json.dumps(proof))


if __name__=='__main__':main()
