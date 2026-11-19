"""Stop one configured capture coordinator; original watchdog restores service.

Never removes evidence, stops the watchdog, rewrites sampling, or retries HTTP.
The separate terminal reconciliation remains mandatory before replacement.
"""
import argparse,json,shlex,subprocess
from pathlib import Path
REMOTE=r'''
import json,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]);r=Path(c['owner']);e=json.loads((r/'EXECUTION.json').read_text())
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['source_boot']
control=Path(e['control_root']);launch=json.loads((control/'NPU-LAUNCH.json').read_text())
info=json.loads(subprocess.check_output(['docker','inspect',c['coordinator']]))[0]
assert info['Id']==launch['container_id'] and info['Name'].lstrip('/')==e['names']['coordinator']
watch=json.loads((control/'WATCHDOG.json').read_text())
w=json.loads(subprocess.check_output(['docker','inspect',watch['container_id']]))[0]
assert w['State']['Running'] or not info['State']['Running']
assert info['HostConfig']['RestartPolicy']['Name']=='no'
assert e['root']==str(r)
if info['State']['Running']:
 subprocess.run(['docker','stop','--time','10',info['Id']],check=True,timeout=30,capture_output=True)
print(json.dumps(dict(coordinator=info['Id'],watchdog=watch['container_id'],
    stop_requested=True,restoration_owner='original_watchdog',automatic_retry=False)))
'''
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());assert c['schema']=='capture-reconfiguration-stop-v1'
out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
if not (out/'STOP-RESULT.json').exists():
 (out/'STOP-INTENT.json').write_text(json.dumps(c,indent=2)+'\n')
 v=subprocess.check_output(['ssh','-o','BatchMode=yes',c['host'],shlex.join(['python3','-B','-c',REMOTE,json.dumps(c)])],text=True,timeout=45)
 (out/'STOP-RESULT.json').write_text(v)
print((out/'STOP-RESULT.json').read_text())
