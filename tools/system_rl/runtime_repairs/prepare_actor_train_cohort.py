"""Prepare the original bounded TRAIN cohort in the sealed source mount namespace."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(config, child):
    deployment=json.loads(config.read_bytes())
    base=Path(deployment['release']).resolve()
    roots=base/'tree/parents'
    output=Path(deployment['output']).resolve()
    if not child:
        from source.release_completion import require_closed_release
        require_closed_release(base)
        if output.exists() and any(output.iterdir()):
            raise ValueError('inspect the original output before preparing again')
        output.mkdir(mode=0o700,parents=True,exist_ok=True)
        logical=Path(json.loads(Path(json.loads((base/'ASSEMBLY.json').read_bytes())['delta_manifest_ref']['path']).read_bytes())['logical_root'])
        # Keep user data/archives read-only and expose only the output for writes.
        subprocess.run(['bwrap','--ro-bind','/','/','--ro-bind',str(roots),str(logical),
                        '--tmpfs','/run','--ro-bind',str(Path(__file__).resolve()),'/run/prepare_actor_train_cohort.py',
                        '--bind',str(output),str(output),'--dev','/dev','--proc','/proc',
                        sys.executable,'-B','/run/prepare_actor_train_cohort.py','--config',str(config),'--child'],check=True)
        return
    sys.path.insert(0,str(roots/'jev-system-rl/implementation'))
    from fresh_adapter import scope
    from run_package import read
    data=read(deployment['train_input'])
    if data['prepare_mode']!='bounded_train_probe' or data['rollout_scope']!='node':
        raise ValueError('bounded node training input required')
    if deployment.get('evidence_input'):
        # Restore the existing base-policy witness for source preparation only.
        # The training selector never consumes diagnostic actions, scores or draws.
        evidence=read(deployment['evidence_input'])['policy']['runtime_evidence_ref']
        witness=dict(evidence,path=deployment['retained_witness_path'])
        read(witness)
        if data['policy']['runtime_evidence_ref'] not in (None,witness):
            raise ValueError('refuse to replace a bound policy witness')
        data['policy']['runtime_evidence_ref']=witness
        data['capture']['endpoint_policy_ref']=witness
    with scope() as cohort:
        sys.path.append(str(roots/'collection-dispatch-integration/production-pipeline-next-version-v1'))
        from bounded_train_probe import install
        install(cohort)
        cohort.prepare(data,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--child',action='store_true')
    a=p.parse_args();run(a.config.resolve(),a.child)
