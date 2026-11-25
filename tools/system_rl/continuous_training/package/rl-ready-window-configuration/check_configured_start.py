"""CPU acceptance for two explicitly supplied real branches; no service or NPU calls."""
import argparse
import json
import sys
import tempfile
from pathlib import Path


def check(branch_path, output):
    from actor_start import compile_start
    from iteration import initial_resume
    from initial_policy import validate_initial
    from run_package import read,pin,write
    from start_experiment import selected_start
    from deploy_resident_graph_lanes import install_configured_policy
    b=read(branch_path);policy=read(b['policy_ref']);publication=read(policy['publication_ref'])
    init=publication['initialization'];node=b['node'];run=publication['lineage']
    actor=compile_start(branch_path,run)
    # Only the initial-state contract is exercised, not a formal sampling roster.
    config=dict(run_id=run,branch_initial_ref=pin(branch_path))
    initial=dict(branch_initial_ref=pin(branch_path),clock_state_ref=b['clock_ref'],learning_config_ref=b['learning_ref'],
                 sampling=dict(scope='node',node=node,split='TRAIN'),native={},
                 policy=dict(current_ref=b['policy_ref'],reference_ref=b['reference_ref']))
    req=dict(target=node,run_id=run,initialization=init,training_input_ref=write(output/'contract-input.json',initial))
    assert selected_start(req,{})==branch_path
    state=initial_resume(config,initial)
    assert state['step']==0 and state['resume_ref'] is None and b['optimizer_inherited'] is False
    def rejected(f):
        try:f()
        except ValueError:return
        raise AssertionError('invalid initialization accepted')
    rejected(lambda:initial_resume(dict(config,initial_resume_ref=init['source_ref']),initial))
    rejected(lambda:selected_start(dict(req,initialization=dict(init,source_ref=dict(init['source_ref'],sha256='0'*64))),{}))
    rejected(lambda:compile_start(branch_path,run+'-other'))
    # Exercise the same bundle projector and validator used in service preflight.
    files=dict(zip(('base.json','profile.json','geometry.json'),(publication[k] for k in ('base_manifest_ref','profile_ref','geometry_ref'))))
    files.update({'source.json':init['source_ref'],'publication.json':policy['publication_ref'],**actor['assets']})
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        src=Path(tmp);area=src/'bundle';area.mkdir()
        c=dict(initial_branch_sha256=actor['branch_ref']['sha256'],configured_files={k:v['sha256'] for k,v in files.items()})
        folder=src/('configured-'+c['initial_branch_sha256']);folder.mkdir()
        for n,r in files.items():(folder/n).symlink_to(Path(r['path']))
        (src/'rank-profile.json').symlink_to(Path(files['profile.json']['path']))
        _,assets=install_configured_policy(src,area,c,pin)
        projected=dict(policy,assets=assets,publication_ref=pin(area/'INITIAL.json'))
        validate_initial(projected)
        assert projected['version']==actor['version'] and [a['sha256'] for a in assets]==[a['sha256'] for a in policy['assets']]
    return dict(mode=init['mode'],node=node,version=actor['version'],source_ref=init['source_ref'],clock=state['step'],
                optimizer_inherited=b['optimizer_inherited'],inference_bundle_verified=True,negative_checks=3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--branch',action='append',required=True,type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();parents=a.release/'tree/parents'
    sys.path[:0]=[str(parents/'jev-system-rl/implementation'),str(parents/'policy-iteration-successor')]
    from source import initial_policy,actor_start,iteration
    sys.modules.update(initial_policy=initial_policy,actor_start=actor_start)
    sys.modules['iteration']=iteration
    from run_package import write
    rows=[check(b,a.output/str(i)) for i,b in enumerate(a.branch)]
    print(json.dumps(write(a.output/'RESULT.json',dict(rows=rows,npu=False,model_calls=0))))
