"""Fixed-image offline verification for a prepared rolling task."""
import argparse,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import load,sha,save
from run import preflight
from budget import receipt
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
require_python='/usr/local/python3.12.13/bin/python3'
assert sys.executable==require_python
start,cpu=time.monotonic(),time.process_time()
tok=AutoTokenizer.from_pretrained('/model',local_files_only=True,trust_remote_code=False)
parent=load('rolling_exact_budget_parent',ROOT/'delivery/s-correct-p-data-v1/code/fit_s_inherited.py')
template=parent.bind_local_chat_template(tok,'/model')
initial=__import__('json').loads(Path(a.config).read_text())
if initial.get('dataset_partition') in ('S_fit','eligible_remainder','S_valid'):
    sys.path.insert(0,str(ROOT/'delivery/s-tfrl-rolling-v1/code'))
    from run_n import preflight
cfg,packets=preflight(a.config,ROOT)
r=receipt(cfg,packets,ROOT,tok);r['chat_template']=template
from run import validate_fit
validate_fit(r,cfg,packets,ROOT)
save(Path(a.output)/'budget.json',r)
if cfg.get('dataset_partition') in ('S_fit','eligible_remainder','S_valid'):
    import subprocess
    dry=Path(a.output)/'CPU-ONLY-BOUND.json'
    save(dry,cfg|dict(runtime=cfg['runtime']|dict(status='OWNER_RECONCILED')))
    sidecar=Path(a.output)/'CPU-ONLY-SIDECAR.json'
    subprocess.run([require_python,'-B',str(ROOT/'delivery/s-tfrl-rolling-v1/code/bind_n.py'),
        '--config',str(dry),'--state',str(Path(a.output)/'empty-cpu-journal'),
        '--output',str(sidecar)],check=True)
    from common import PACKAGE
    assert __import__('json').loads(sidecar.read_text())['branch_adapter_sha256']==sha((PACKAGE/'code/run.py').read_bytes())
    save(Path(a.output)/'BIND-CPU-REPLAY.json',dict(status='PASS',simulated_runtime_only=True,
        guard_acquired=False,lane_acquired=False,service_contact=False,model_calls=0))
save(Path(a.output)/'CPU-VERIFIED.json',dict(status='PASS_FIXED_IMAGE_CPU',image=r['image'],rows=len(packets),samples=cfg.get('full_denominator',545),
     manifest_sha256=cfg['manifest_sha256'],snapshot_sha256=cfg['snapshot_sha256'],max_total_upper=r['max_total_upper'],
     audio_encodes=0,model_calls=0,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
     script_sha256=sha(Path(__file__).read_bytes()),actual_tokenizer=True,validate_fit_replayed=True))
print(json.dumps(dict(status=r['status'],rows=len(packets),max_total_upper=r['max_total_upper'])))
