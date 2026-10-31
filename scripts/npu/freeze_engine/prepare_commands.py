"""Derive the owner's frozen deployment argv in the formal CPU container."""
import argparse,json
from pathlib import Path
from speech_aware_evidence_acquisition.contracts import assert_execution_environment
from speech_aware_evidence_acquisition.core.deployment import load_deployment_profile,model_fragment_values,docker_run_argv,allocate_from_npu_smi
assert_execution_environment()
S=Path('/mnt/cephfs/xingchao/omni/publish/omni/studies/speech-aware-evidence-acquisition')
p=argparse.ArgumentParser();p.add_argument('--nic',required=True);p.add_argument('--smi');p.add_argument('--profile',default='qwen3-omni-thinking-bf16-4xtp2-s48-b2048-lock-v1');a=p.parse_args()
profile=load_deployment_profile(a.profile,S)
if a.smi:allocate_from_npu_smi(profile,Path(a.smi).read_text())
out=[]
for instance in profile.instances:
 out.append({'name':instance.name,'host':profile.host,'port':instance.port,'cards':list(instance.cards),'profile_id':profile.profile_id,'profile_sha256':profile.sha256,'argv':docker_run_argv(profile,instance,model_fragment_values(profile,S),nic=a.nic)})
print(json.dumps(out))
