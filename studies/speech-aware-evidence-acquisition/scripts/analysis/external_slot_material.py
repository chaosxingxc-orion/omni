"""Prepare private reviewed slot-store material from an explicit configuration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from external_slot_candidate import read
from external_slot_workload import measure


def prepare(config):
    cfg=read(config);repo=Path(cfg['repo']);data=Path(cfg['data']);root=Path(cfg['root']);inputs=Path(cfg['input'])
    if root.exists() and any(root.iterdir()):raise ValueError('fresh material output required')
    root.mkdir(parents=True,exist_ok=True)
    def write(p,d):
        with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    def pin(p):return {'relpath':str(p.relative_to(data)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    index=cfg['candidate_index'];candidate=read(inputs/f'candidate-{index}/candidate.json')
    write(root/'candidate.json',candidate);write(root/'snapshot.json',candidate['snapshot'])
    write(root/'compiler.json',read(inputs/f'compile-{index}.json'));write(root/'review.json',read(inputs/'ROOT-REVIEW.json'))
    snapshot=candidate['snapshot'];write(root/'release.json',{'schema':'saea-knowledge-store-release-v1','store_id':'slot-contracts','store_role':'slot-contracts','snapshot_sha256':snapshot['snapshot_sha256'],'source_split_identity_hash':snapshot['source_split_identity_hash'],'builder_policy_id':snapshot['builder_policy_id'],'source':pin(root/'candidate.json')})
    snapshot_pin={'relpath':str((root/'snapshot.json').relative_to(data)),'sha256':snapshot['snapshot_sha256']}
    measured=measure(repo,data,candidate,cfg['runtime_binding'],snapshot_pin);write(root/'measurement.json',measured)
    material={'schema':'saea-external-slot-knowledge-material-v1',**{key:pin(root/(name+'.json')) for key,name in [('compiler','compiler'),('candidate','candidate'),('release','release'),('review','review'),('measurement','measurement')]}}
    path=repo/cfg['material_relpath'];path.parent.mkdir(parents=True,exist_ok=True);write(path,material)
    return {'material':{'relpath':cfg['material_relpath'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'new_model_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args();print(json.dumps(prepare(a.config)))
