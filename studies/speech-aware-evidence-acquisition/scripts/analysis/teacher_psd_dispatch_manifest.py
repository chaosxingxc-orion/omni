"""Validate compiled teacher PSD candidates and emit CPU/NPU dispatch commands."""
import argparse, hashlib, json
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--data-root',default='/mnt/cephfs/xingchao/omni');p.add_argument('--tree',default='/mnt/cephfs/xingchao/omni/work/dev/odd40-runtime-v1');a=p.parse_args()
    items=[]
    for path in sorted(a.configs.glob('*.json')):
        d=json.loads(path.read_text())
        if len(d.get('sample_ids',[])) != 1080 or d.get('node_arm') != 'enhanced-p-slot-decision': raise ValueError(f'invalid candidate {path}')
        node='decision' if 'decision_policy' in d else 'slot'
        alg='ace' if d['name'].startswith('slurp-teacher-ace-') else 'reflexion'
        if alg=='ace' and node!='slot': raise ValueError('ACE Decision candidate forbidden')
        name=d['name']; study=f'{a.tree}/studies/speech-aware-evidence-acquisition'; data=a.data_root
        stage=f'{data}/work/codex/teacher-psd-candidate-integration-v1/stage/{name}'
        prep=f'{data}/work/codex/teacher-psd-candidate-integration-v1/prepare/{name}'
        prefix=f'TREE={a.tree} bash {data}/work/scripts/port_py.sh'
        items.append({'name':name,'algorithm':alg,'target_node':node,'fold': 'B' if 'fold-b' in name else 'A','rows':1080,'cards':4,'topology':'TP4','model_fragment':d['runtime_binding']['model_fragment'],'profile':d['runtime_binding']['profile'],'config_relpath':str(path),'config_sha256':sha(path),'new_model_calls':0,'generation_authorized':False,'commands':{'stage':f'{prefix} scripts/analysis/intent_slot_workload.py stage --repo {study} --data {data} --config {study}/{path.relative_to(a.configs.parent.parent.parent)} --out {stage}','prepare':f'{prefix} scripts/analysis/intent_slot_workload.py prepare --repo {study} --data {data} --config {study}/{path.relative_to(a.configs.parent.parent.parent)} --stage-record {stage}/complete.json --out {prep}'}})
    out={'schema':'saea-teacher-psd-dispatch-ready-v1','status':'cpu-compiled-ready-npu-admission-pending','source':'recorded collected responses only','teacher_calls':0,'npu_tasks_started':0,'monitor_interval_seconds':600,'candidates':items}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
