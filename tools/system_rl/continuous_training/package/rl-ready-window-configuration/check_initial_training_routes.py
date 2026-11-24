"""Consume an actual prepared window and its activated residents without sending requests."""
import argparse
from pathlib import Path
import json

def check(cohort, module, output):
    from dependencies import load
    from run_package import read,pin,write
    from fresh_adapter import scope
    cfg=read(cohort/'INPUT.json');plan=read(cohort/'SELECTED.json')
    with scope():
        from policy import validate
        current,_=validate(cfg['policy'],read(cfg['learning_config_ref']),read(cohort/'RESERVED-CLOCK.json'))
        driver=load(module,'initial_routes')
        capture=cfg['capture'];routes=driver.admitted_routes(current,capture['endpoint_policy_ref'],read(capture['profile_ref']),True)
        assignments=driver.lane_plan(plan,capture,routes)
    return write(output,dict(window=plan['sampled_window_ref'],driver_ref=pin(module),policy=cfg['policy']['current_ref'],
        lanes={key:list(assignments.values()).count(key) for key in routes},dispatch=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('cohort','module','output'):p.add_argument('--'+n,type=Path,required=True)
    print(json.dumps(check(**vars(p.parse_args()))))
