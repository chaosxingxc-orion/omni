"""Archive full two-arm results after coordinator terminal; retain every failure."""
import importlib.util,json,os,subprocess,sys,time,traceback,fcntl
from pathlib import Path
W=Path(__file__).resolve().parents[3];O=W/'delivery/s-single-intent-execution-v1';H='/srv/omni-local-state/s-single-intent-execution-v1';CID='477b57cb9e723e7037818ea6a13e42c5eba700dd927feda16b9a813ee040ee4e'
spec=importlib.util.spec_from_file_location('paired_archive_owner',O/'code/collector_local.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
guard=(O/'paired-collector.guard').open('a+b');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);module.save(O/'receipts/PAIRED-COLLECTOR-START.json',dict(pid=os.getpid(),at=time.time(),coordinator=CID,model_calls=0,trigger='exact coordinator docker wait'))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','npu','docker wait '+CID],capture_output=True);module.save(O/'receipts/PAIRED-WAIT.json',dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),at=time.time()))
while True:
 try:
  for lane in ['C-v2','D']:
   launch=json.loads(module.ssh(['cat',H+'/'+lane+'/LAUNCH.json'],capture_output=True).stdout);(O/'results'/lane).mkdir(parents=True,exist_ok=True)
   with (O/'results'/lane/'container.log').open('wb') as f:module.ssh(['docker','logs',launch['container_id']],stdout=f,stderr=subprocess.STDOUT);f.flush();os.fsync(f.fileno())
  path=module.capture('paired-terminal-'+str(time.time_ns()));result=json.loads((path/'PAIR-TERMINAL.json').read_bytes());module.save(O/'TWO-ARM-RESULT.json',result);module.save(O/'receipts/PAIR-ARCHIVE.json',dict(path=str(path.relative_to(W)),at=time.time(),model_calls=0))
  text='S 单意图两臂终态已归档。\n\n'
  if result['status']=='COMPLETE_TWO_ARM_PAIRED_SCORED':
   pair=result['paired'];text+='两臂各545样本/1635真实分支，原score_s、多槽multiset、局部/整行失败均分别保留；官方条件槽位不是D/E2E。\n\n'+json.dumps(pair,ensure_ascii=False,indent=2)+'\n'
  text+='\n完整结果见 TWO-ARM-RESULT.json；旧HTTP400无模型生成，0.030761秒及全部历史成本保留。货币未知。\n';(O/'REPORT.md').write_text(text,encoding='utf-8');module.save(O/'READY.json',dict(status=result['status'],at=time.time(),archive=str(path.relative_to(W)),model_calls=0));break
 except Exception:
  module.save(O/'receipts'/('PAIR-ARCHIVE-ERROR-'+str(time.time_ns())+'.json'),dict(error=traceback.format_exc(),at=time.time()));time.sleep(600)
