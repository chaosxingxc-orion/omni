"""Replay closed CPU evidence through the original policy validator. No model requests."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


def replay(config):
    spec=json.loads(config.read_bytes())
    run=Path(spec['iteration_ref']['path']).parent.parent/'run'
    sys.path.insert(0,str(config.parent))
    from transition_controller import main
    previous=sys.argv
    try:
        sys.argv=['transition_controller.py',str(config),'--check']
        main()
    finally:sys.argv=previous
    from dependencies import register_locations
    from run_package import read
    from fresh_adapter import scope
    from transition_static_inputs import for_window
    for p in run.glob('step-*/update/returned/LOCATION-MAP.json'):register_locations(read(p))
    def rejected(fn,message):
        try:fn()
        except ValueError:return True
        raise ValueError(message)
    checks={};groups=0;windows=0
    with scope() as cohort:
        import policy, publication
        for d in sorted((run/'rollout').glob('*-capture/cohort')):
            if not (d/'GENERATION-CLOSED.json').is_file():continue
            cfg=read(d/'INPUT.json');plan=read(d/'SELECTED.json')
            cfg['capture'].update({k:spec[k] for k in ('static_supply_ref','prompt_length_index_ref')})
            cached=for_window(cfg,plan)
            if [r['annotation'] for r in cached]!=read(d/'ANNOTATION-BATCH.json')['rows']:
                raise ValueError('annotations differ')
            groups+=len(cached);windows+=1
        if not groups:raise ValueError('no closed window to test')
        bad=copy.deepcopy(plan);bad['draws'][0]['record']['source_id']='wrong'
        checks['wrong_record']=rejected(lambda:for_window(cfg,bad),'accepted changed record')
        clock=cohort.clock_for(cfg);n=[];old=publication.artifact
        def counted(*a,**kw):n.append(1);return old(*a,**kw)
        publication.artifact=counted
        def val(state):return policy.validate(cfg['policy'],read(cfg['learning_config_ref']),state)
        try:
            t=time.monotonic();a=val(clock.state);cold=time.monotonic()-t
            t=time.monotonic();b=val(clock.state);warm=time.monotonic()-t
            wrong=copy.deepcopy(clock.state);wrong['successful_optimizer_steps']+=1
            checks['wrong_clock']=rejected(lambda:val(wrong),'accepted changed Clock')
        finally:publication.artifact=old
        if a!=b or len(n)!=1:raise ValueError('unchanged checkpoint did not reuse one validation')
    return dict(passed=True,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                replay=dict(groups=groups,windows=windows,cold=cold,warm=warm,artifact_reads=len(n),negative=checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();value=replay(args.config.resolve())
    with args.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value))
