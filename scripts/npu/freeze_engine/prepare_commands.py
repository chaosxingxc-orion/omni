"""Derive the owner's frozen deployment argv in the formal CPU container."""
import argparse,json
from pathlib import Path
from speech_aware_evidence_acquisition.contracts import assert_execution_environment
from speech_aware_evidence_acquisition.core.deployment import load_deployment_profile,model_fragment_values,docker_run_argv,allocate_from_npu_smi,CardRefusal
assert_execution_environment()
S=Path('/mnt/cephfs/xingchao/omni/publish/omni/studies/speech-aware-evidence-acquisition')
p=argparse.ArgumentParser();p.add_argument('--nic',required=True);p.add_argument('--smi');p.add_argument('--release-status',action='store_true');p.add_argument('--profile',default='qwen3-omni-thinking-bf16-4xtp2-s48-b2048-lock-v1');a=p.parse_args()
profile=load_deployment_profile(a.profile,S)
if a.smi:
    try:allocate_from_npu_smi(profile,Path(a.smi).read_text())
    except CardRefusal as error:
        if not a.release_status:raise
        reasons=[reason for group in error.refusals.values() for reason in group]
        print(json.dumps({'ready':False,'memory_release_only':bool(reasons) and all(r.startswith('HBM used ') for r in reasons),'refusals':error.refusals}));raise SystemExit(0)
    if a.release_status:print(json.dumps({'ready':True,'memory_release_only':False}));raise SystemExit(0)
out=[]
for instance in profile.instances:
 out.append({'name':instance.name,'host':profile.host,'port':instance.port,'cards':list(instance.cards),'profile_id':profile.profile_id,'profile_sha256':profile.sha256,'argv':docker_run_argv(profile,instance,model_fragment_values(profile,S),nic=a.nic)})
print(json.dumps(out))
