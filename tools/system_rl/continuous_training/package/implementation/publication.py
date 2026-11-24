"""Closed DP8 checkpoint -> original policy manifests; no new trainer or RPC."""
import argparse
import json
from pathlib import Path
from dependencies import require,sha,HERE,local_path
from policy_identity import artifact,digest
from run_package import pin,read,write


def checkpoint(ref,*,verify_shards=False):
    value=read(ref);logical_root=Path(pin(ref)['path']).parent;root=local_path(pin(ref)['path']).parent
    require(value['world_size']==8 and len(value['ranks'])==8 and {r['rank'] for r in value['ranks']}==set(range(8)),'actual DP8 rank commit required')
    require(value['step']==value['clock']['successful_optimizer_steps']>0,'checkpoint successful step differs')
    if value.get('kind')=='lora_training_state':
        from lora_training_state import required_files
        required_files(value)
    files={}
    for row in value['ranks']:
        for name,expected in row['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'checkpoint path escapes root')
            path=local_path(logical_root/name).resolve()
            if verify_shards or name.startswith('lora_adapter/'):
                require(sha(path)==expected,'checkpoint shard changed: '+name)
                files[name]=dict(pin(path),path=str(logical_root/name))
    adapter=local_path(logical_root/'lora_adapter/adapter_model.safetensors')
    require('lora_adapter/adapter_model.safetensors' in files and 'lora_adapter/adapter_config.json' in files,'original adapter export absent')
    observed=artifact(adapter)
    require(observed==value['resident_identity']['publication'] and observed['nonzero_effective_delta'],'actual nonzero exported adapter differs from resident')
    ac=read(logical_root/'lora_adapter/adapter_config.json')
    require(ac['r']==8 and ac['lora_alpha']==16 and ac['lora_dropout']==0 and ac['bias']=='none' and not ac.get('use_dora',False),'adapter config differs')
    return value,files


def seal(checkpoint_ref,output,*,fresh_commit=False):
    # The rank writers just hashed, synced and committed their local shard
    # bytes. A live command publishes only the adapter verified by checkpoint().
    value,files=checkpoint(checkpoint_ref,verify_shards=not fresh_commit);output=Path(output)
    # Candidate is loadable for updated parity before promotion. A service
    # activation is a physical endpoint witness, not a scientific promotion.
    assets=[files['lora_adapter/'+n] for n in ('adapter_model.safetensors','adapter_config.json')]
    version='checkpoint-sha256:'+assets[0]['sha256']
    pub=write(output/'POLICY-CANDIDATE.json',dict(schema='omni-dp8-publication-v1',status='candidate_for_updated_parity',
        checkpoint_ref=pin(checkpoint_ref),step=value['step'],clock=value['clock'],assets=assets,
        artifact_version=version,resident_identity=value['resident_identity'],scientific_contract=value['contract']))
    manifest=write(output/'ASSETS.json',dict(status='CLOSED_IMMUTABLE',assets=assets,publication_ref=pub))
    return write(output/'POLICY.json',dict(schema='fresh-policy-version-v1',role='current',kind='checkpoint',
        version=version,artifact_version=version,step=value['step'],model='qwen3-omni-30b-a3b-instruct',thinking=False,
        assets=assets,asset_manifest_ref=manifest,publication_ref=pub))


def validate_publication(ref, *, verify_parity=True,verify_checkpoint_shards=False):
    value=read(ref)
    if value.get('schema')!='omni-dp8-publication-v1':
        # Preserve historical four-rank evidence under the original verifier.
        from dependencies import ROOT,load
        import sys
        sys.path.insert(0,str(ROOT/'fresh-cohort-successor'))
        return load(ROOT/'fresh-cohort-successor/policy.py','dp8_historical_policy').validate_publication(ref)
    committed,files=checkpoint(value['checkpoint_ref'],verify_shards=verify_checkpoint_shards)
    require(value['step']==committed['step'] and value['clock']==committed['clock'] and value['resident_identity']==committed['resident_identity'],'publication checkpoint drift')
    require(value['assets']==[files['lora_adapter/'+n] for n in ('adapter_model.safetensors','adapter_config.json')],'publication assets differ')
    require(value['artifact_version']=='checkpoint-sha256:'+value['assets'][0]['sha256'],'publication version differs')
    require(value['status'] in ('candidate_for_updated_parity','published_for_fresh_sampling'),'unknown publication state')
    if value['status']=='published_for_fresh_sampling' and verify_parity:
        if 'continuous_activation_ref' in value:
            from continuous_policy import verify
            verify(value,value['continuous_activation_ref'])
        elif 'bounded_tp4_ref' in value:
            verify_bounded(value, value['bounded_tp4_ref'])
        else:
            from parity_gate import verify
            runtime=read(value['runtime_ref'])
            verify(value['updated_rcp_ref'],config=runtime,policy_version=value['artifact_version'],
                   source_pins_sha256=sha(HERE/'SOURCE-PINS.json'),require_updated=True,identity=value['resident_identity'])
    return value


def verify_bounded(pub, ref):
    """Replay the scoped P same-action evidence; no formal domain admission."""
    from bounded_tp4 import replay
    return replay(pub, ref)


def promote(policy_ref,raw_ref,runtime_ref,output,*,bounded_tp4=False):
    policy=read(policy_ref);pub=validate_publication(policy['publication_ref'])
    if bounded_tp4:
        require(runtime_ref is None,'bounded replay takes its sealed window config')
        verify_bounded(pub,pin(raw_ref))
        result=dict(pub,status='published_for_fresh_sampling',bounded_tp4_ref=pin(raw_ref))
    else:
        from parity_gate import verify
        verify(pin(raw_ref),config=read(runtime_ref),policy_version=pub['artifact_version'],
               source_pins_sha256=sha(HERE/'SOURCE-PINS.json'),require_updated=True,identity=pub['resident_identity'])
        result=dict(pub,status='published_for_fresh_sampling',updated_rcp_ref=pin(raw_ref),runtime_ref=pin(runtime_ref))
    promoted=write(Path(output)/'POLICY-PUBLISHED.json',result)
    return write(Path(output)/'POLICY.json',dict(policy,publication_ref=promoted))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(seal(a.checkpoint,a.output)))
if __name__=='__main__':main()
