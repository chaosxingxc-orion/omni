"""Replay actual checkpoint/generation admission without running the actor or NPU."""
import argparse,copy,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for k in ('run','release'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--steps',type=int,nargs='+',required=True)
    a=p.parse_args();home=Path(__file__).parent
    sys.path[:0]=[str(home),str(a.release/'tree/parents/jev-system-rl/implementation')]
    from run_package import pin,read
    from dependencies import register_locations
    from dependencies import load
    formal_window=load(home/'source/resident.py','candidate_resident').formal_window
    from bounded_actor_step import verify_continuation
    initial=None;previous=None;observed=[];rejected=0
    def reject(fn):
        nonlocal rejected
        try:fn()
        except ValueError:rejected+=1
        else:raise AssertionError('wrong live state admitted')
    for index,step in enumerate(a.steps):
        out=a.run/f'step-{step:06d}'/'update';prev=a.run/f'step-{step-1:06d}'/'update'/'returned'
        register_locations(read(prev/'LOCATION-MAP.json'));register_locations(read(out/'returned/LOCATION-MAP.json'))
        last=read(prev/'RETURN.json')['checkpoint_ref'];prior=read(last)
        produced=read(out/'returned/RETURN.json')['checkpoint_ref'];after=read(produced)
        assert prior['step']==step-1 and after['step']==step
        if previous is not None:assert last==previous
        req=read(next(out.glob('TRAIN-REQUEST*.json')));gen=req['generation']
        fields=after['contract'];sources=fields['sources']
        if initial is None:
            initial=dict(name='offline',lineage=fields['lineage'],formal_sampled=True,deterministic_backward=True,
                training=fields['training'],devices=fields['devices'],image=fields['image'],
                source_pins=sources,files=dict(sources,finalized_generation=fields['accepted_fields_sha256'],
                    **{'lora_row_precision.py':fields['lora_arithmetic']['source_sha256']}),
                lora_row_geometry=fields['lora_arithmetic']['geometry'],accepted_fields='finalized_generation',
                dataset_sha256=fields['dataset_sha256'],continuation=dict(rollout_lanes=req['rollout_lanes']),generation=gen)
        cmd=dict(action='update',run_id='offline',sequence=index,previous_close_sha256='offline',checkpoint_sha256=last['sha256'],
                 pending_capture_ref=gen['pending_capture_ref'],generation_inventory_ref=gen['generation_inventory_ref'])
        spec=formal_window(initial,cmd,last,prior,[])
        real=read(read(out/'returned/RETURN.json')['result_ref'])
        pending=read(gen['pending_capture_ref'])
        clock=read(read(pending['pending_capture_ref'])['cohort_root']+'/RESERVED-CLOCK.json')
        packet=dict(clock=clock,learning=read(pending['learning_config_ref']),selected_groups=real['group_ids'],attempt_id=real['attempt_id'])
        spec.update(dataset_sha256=fields['dataset_sha256'],files=dict(spec['files'],finalized_generation=fields['accepted_fields_sha256']))
        verify_continuation(spec,prior,packet)
        assert len(packet['selected_groups'])==32 and real['result']['objective_observation']['global_episode_denominator']==128
        reject(lambda:formal_window(initial,dict(cmd,checkpoint_sha256=produced['sha256']),last,prior,[]))
        reject(lambda:verify_continuation(spec,prior,dict(packet,attempt_id='stale')))
        if previous is not None:
            reject(lambda:formal_window(initial,dict(cmd,generation_inventory_ref=previous_gen),last,prior,[]))
        previous_gen=gen['generation_inventory_ref'];previous=produced;observed.append(step)
    print(json.dumps(dict(steps=observed,invalid_commands_rejected=rejected,scope='offline actual checkpoint, sampled input and Clock admission; no new training')))
if __name__=='__main__':main()
