"""Close a restored, pre-reservation capture failure without replaying it."""
import argparse,base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path

REMOTE=r'''
import base64,hashlib,json,sqlite3,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]);plan=c['plan'];job=plan['job'];root=Path(plan['target'])
def read(p):return json.loads(p.read_text())
def inspect(cid):
 v=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
 assert not v['State']['Running'] and v['State']['Status']=='exited'
 return v
controller=inspect(c['controller'])
assert controller['State']['ExitCode']!=0 and read(root/'FAILED.json')['error']=='OWNER_STAGE_UNKNOWN_RECONCILE_NO_REPLAY'
owner=Path(c['owner']);control=owner/'hot/new-lease'
actor=inspect(read(control/'NPU-LAUNCH.json')['container_id'])
watch=inspect(read(control/'WATCHDOG.json')['container_id'])
assert actor['State']==read(control/'NPU-EXIT.json') and actor['State']['ExitCode']!=0
assert watch['State']['ExitCode']==0
logs=subprocess.check_output(['docker','logs',actor['Id']],stderr=subprocess.STDOUT)
cohort=owner.parent/'cohort'
reserved=(cohort/'RESERVED-CLOCK.json').exists()
assert (cohort/'SELECTED.json').exists()
reconfigured=c.get('reason')=='sampling_scope_reconfiguration'
if not reconfigured:
 assert b'ValueError: only designated unified implementation supported' in logs
 assert b'clock=clock_for(config)' in logs and b'cohort.prepare(config,root)' in logs
 assert not reserved and not (cohort/'PENDING-CAPTURE.json').exists() and not (cohort/'groups').exists()
else:
 assert reserved and actor['Id']==c['stop']['coordinator']
 assert c['stop']['stop_requested'] and c['stop']['automatic_retry'] is False

restored=read(control/'RESTORED.json');assert restored['config_exact'] is True
service=json.loads(subprocess.check_output(['docker','inspect',restored['container_id']]))[0]
assert service['State']['Running']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==job['source_boot']
spec=job['accounting'];path=Path(spec['remote_parent'])/spec['basename']
with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
 facts={t:db.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall() for t in ('meta','history','spend','samples','batches')}
assert all(r[2] in ('known','closed_unobserved') for r in facts['spend'])
for table in facts:assert all(tuple(r) in facts[table] for r in spec['before'][table])
backup=root/'failed-capture-accounting.sqlite'
if not backup.exists():
 with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(backup) as dst:src.backup(dst)
with sqlite3.connect(backup.as_uri()+'?mode=ro',uri=True) as db:
 assert facts=={t:db.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall() for t in facts}
raw=backup.read_bytes()
print(json.dumps(dict(schema='failed-capture-terminal-proof-v1',source_boot=job['source_boot'],
 controller=dict(id=controller['Id'],state=controller['State']),actor=dict(id=actor['Id'],state=actor['State']),
 watchdog=dict(id=watch['Id'],state=watch['State']),restored=restored,all_owners_terminal=True,
 optimizer_updates=0,cohort_reserved=reserved,
 cohort_requests=sum(1 for r in facts['spend'] if r[0] not in {x[0] for x in spec['before']['spend']}),
 reconfiguration=reconfigured,
 failure_logs=logs.decode(errors='replace'),failure_log_sha256=hashlib.sha256(logs).hexdigest(),
 before=spec['before'],after=facts,database_b64=base64.b64encode(raw).decode(),database_sha256=hashlib.sha256(raw).hexdigest())))
'''

def main():
 p=argparse.ArgumentParser()
 for n in ('release','config'):p.add_argument('--'+n,required=True)
 a=p.parse_args();sys.path.insert(0,str(Path(a.release)/'tree/parents/jev-system-rl/implementation'))
 from run_package import read,pin,write
 from registry_custody import accept_closed
 from training_handoff import terminal
 c=read(a.config);assert c['schema'] in ('failed-capture-reconciliation-v1','stopped-capture-reconciliation-v1')
 terminal(c['client']);plan=read(c['transport_ref']);old=read(c['iteration_ref'])
 out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=True)
 proofpath=out/'TERMINAL-PROOF.json'
 if not proofpath.exists():
  result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',plan['host'],
   shlex.join(['python3','-B','-c',REMOTE,json.dumps(dict(plan={'target':plan['target'],'job':{k:plan['job'][k] for k in ('source_boot','accounting')}},controller=c['controller'],owner=c['owner'],reason=c.get('reason'),stop=read(c['stop_ref']) if c.get('stop_ref') else None))])],text=True,timeout=60)
  result=json.loads(result);raw=base64.b64decode(result.pop('database_b64'))
  assert hashlib.sha256(raw).hexdigest()==result['database_sha256']
  db=out/'REGISTRY.sqlite'
  if db.exists():assert db.read_bytes()==raw
  else:
   import os
   with db.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  result['database_ref']=pin(db);result['config_ref']=pin(a.config)
  write(proofpath,result)
 proof=read(proofpath);assert proof['config_ref']==pin(a.config)
 closed=write(out/'ACCOUNTING-CLOSED.json',dict(immutable=True,before=proof['before'],after=proof['after'],database_ref=proof['database_ref']))
 accept_closed(old['registry'],closed,out)
 print(json.dumps(dict(proof_ref=pin(proofpath),accounting_accepted=True,dispatch=False)))

if __name__=='__main__':main()
