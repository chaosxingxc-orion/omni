"""Use local persistent root disk for hot fsync, then archive once to delivery."""
import json,os,subprocess,sys,time,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,save
name=sys.argv[1];seed=sys.argv[2] if len(sys.argv)>2 else None
hot=Path('/var/tmp/omni-s-tfrl-rolling-v1')
kind='grpo-reserve' if seed else 'candidates'
env=os.environ.copy();env['S_ROLLING_CPU_OUTPUT']=str(hot)
args=[sys.executable,'-B',str(OUT/'code/prepare_followups.py'),name]+(['--seed',seed] if seed else [])
subprocess.run(args,env=env,check=True)
source=hot/kind/name;target=OUT/kind/name
assert not target.exists(),'partial attempt must be archived before publishing'
pending=target.with_name(target.name+'.pending');pending.mkdir(parents=True)
archive=hot/(name+'.tar')
with tarfile.open(archive,'w') as tf:
 for p in source.rglob('*'):
  if p.is_file():tf.add(p,arcname=str(p.relative_to(source)),recursive=False)
with archive.open('rb') as f:subprocess.run(['tar','xf','-','-C',str(pending)],stdin=f,check=True)
# syncfs preserves durable delivery after bulk transfer; original hot files each fsynced.
subprocess.run(['sync','-f',str(pending)],check=True)
os.rename(pending,target)
save(target/'ARCHIVE-RECEIPT.json',dict(at=time.time(),hot=str(source),durability='hot per-file fsync + delivery syncfs + directory rename',model_calls=0))
