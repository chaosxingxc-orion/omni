"""Verify pre-reference rewards and advantages against the unmodified learner."""
from pathlib import Path
import argparse,json,sys,os
def check(root):
    from bounded_actor_step import generation_sources
    from fresh_adapter import scope,consumer
    from dependencies import load,require
    from microbatch_roster import group_rewards
    read=lambda p:json.loads(p.read_bytes())
    with generation_sources(read(Path('/sealed/TRAIN.local.json'))) as rt:
        os.environ['OMNI_CAPTURE_CONSUMER_SOURCE']=rt['consumer_source']
        os.environ['OMNI_CAPTURE_CONSUMER_WRAPPER']=rt['consumer_wrapper']
        with scope():
            from cohort_io import read as registered,pin
            scorer=load(rt['capture_source'],'microbatch_reward_capture')
            cons=consumer(load(rt['consumer_wrapper'],'microbatch_reward_consumer'))
            cfg=registered(pin(root/'LOCAL-CONSUMER.json')); lc=registered(cfg['learning_config_ref'])
            plan=registered(pin(root/'SELECTED.json'));clock=registered(cfg['clock_state_ref'])
            rep=registered(pin(root/'CONSUMER-REPORT.json'))
            expected={(g['group_id'],e['episode_id']):e for g in rep['group_results'] for e in g['episodes']}
            produced=[]
            with cons.unified(cfg) as (learning,schedule,kernel):
                for d in plan['draws']:
                    group=registered(pin(root/'groups'/str(d['draw_index'])/'GENERATED.json'))
                    require(plan['policy']==group['policy'],'different policy')
                    snap=clock['groups'][d['group_id']]['snapshot']
                    produced.extend(group_rewards(d,group,snap,lc,labels_for=scorer.labels_for,
                        score_episode=learning.evaluate.__globals__['score_episode'],reward=learning.reward,
                        advantages=learning.evaluate.__globals__['reduce_validated'].__globals__['advantages']))
            require(len(expected)==len(produced)==rep['global_episode_denominator'],'wrong episode roster')
            require(json.dumps([{k:v for k,v in row.items() if k!='advantage'} for row in produced],sort_keys=True,allow_nan=False)==json.dumps(rep['component_scores'],sort_keys=True,allow_nan=False),'different score receipt')
            for r in produced:
                e=expected[r['group_id'],r['episode_id']]
                require((e['advantage'],e['reward'])==(r['advantage'],r['reward']),'different advantage')
            return dict(groups=len(plan['draws']),episodes=len(produced),component_scores_equal=True,
                        advantages_equal=True,reference_scores_consumed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path,action='append');a=p.parse_args()
    sys.path.insert(1,'/sealed');print(json.dumps([check(root) for root in a.root]))
