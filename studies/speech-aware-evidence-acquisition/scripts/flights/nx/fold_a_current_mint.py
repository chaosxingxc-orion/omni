"""Mint only the independently registered current-A0 Fold A decision."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'direct'))
import fold_a_current_registration as registration
from l338_port_mint_receipt import stale_cells
from thinking_system_mint_receipt import mint


def admit(repo,data,dry=False):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('committed clean registration required')
    v=registration.evaluation;draft=registration.read(repo/registration.PKG/'root-admission-draft-v1.json')
    runs=[v.RUN,v.FIT]
    if (draft['status']!='DRAFT-NOT-MINTED' or draft['eligible_run_ids']!=runs
            or draft['eligible_attempt_ids']!=['generation-v1','fit-v1']):raise ValueError('Fold A admission identity differs')
    bound=draft['budget']['accelerator_bound']
    binding=registration.validate_binding(bound['measurement'],repo,data)
    _,pins=registration.accepted_proof(repo,data,binding)
    if bound['status']!='MEASURED-SYNTHETIC-ALLOWANCE' or pins!=bound['proof_inputs']:raise ValueError('accepted proof changed')
    # Never allow generic mint's rehash to silently bless changed antecedents.
    def verify(node):
        if isinstance(node,dict):
            if 'path' in node and 'sha256' in node:
                path=v.comparator.inputs.confined(repo,node['path'])
                if registration.sha(path)!=node['sha256']:raise ValueError('registered antecedent changed')
            for value in node.values():verify(value)
        elif isinstance(node,list):
            for value in node:verify(value)
    verify(draft['antecedents'])
    family=SimpleNamespace(PACKAGE=registration.PKG,DECISION_ID=v.DECISION,DECISION_RELPATH=v.DECISION_PATH,
        ALL_KEYS=['arm'],CELLS={'arm':{'cell':v.CELL}},eligible_run_ids=lambda:runs)
    problems=stale_cells(repo,family)
    if problems:raise ValueError('stale Fold A registration: '+'; '.join(problems))
    return mint(str(repo),str(data),'Codex',dry,family=family,tool='scripts/flights/nx/fold_a_current_mint.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dry',action='store_true');a=p.parse_args()
    print(json.dumps(admit(a.repo_root,a.data_root,a.dry),sort_keys=True))


if __name__=='__main__':main()
