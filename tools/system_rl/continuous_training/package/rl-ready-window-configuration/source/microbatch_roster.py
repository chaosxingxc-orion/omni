"""Derive policy microbatch rows from a sealed rollout, before reference scoring."""
from events import assign_policy_rows
from dependencies import require


def roster(draws, generated, *, world, micro, accumulation, trainable_mask, group_size, policy):
    require(type(group_size) is int and group_size>0,'invalid generation group size')
    rows=[]; episodes=set(); requests=set();groups=set()
    for draw,group in zip(draws,generated,strict=True):
        gid=draw['group_id']
        require(group['policy']==policy and group['group_id']==gid and gid not in groups and
                group['query_draw_id']==draw['query_draw_id'],'wrong or duplicate draw')
        groups.add(gid)
        require([e['replicate'] for e in group['episodes']]==list(range(group_size)),'incomplete group')
        for ep in group['episodes']:
            eid=ep['episode_id'];require(eid not in episodes and not ep['synthetic'],'different episode')
            episodes.add(eid)
            for c in ep['calls']:
                packet=c['packet'];node=packet['stage'];rid=packet['request_id'];identity=packet['fresh_capture_identity']
                require(rid not in requests and identity['episode_id']==eid and identity['group_id']==gid
                        and identity['replicate']==ep['replicate'] and identity['node']==node,'wrong request')
                requests.add(rid)
                if trainable_mask[node] and c['token_ids']:
                    require(c['prompt_token_ids'] and len(c['token_ids'])==len(c['behavior_logprobs']),'missing retained tokens')
                    rows.append((gid,eid,rid,len(c['prompt_token_ids'])+len(c['token_ids'])))
    owners,loads=assign_policy_rows([r[3] for r in rows],world,micro,accumulation)
    by_rank=[[i for i,o in enumerate(owners) if o==rank] for rank in range(world)]
    slots=max(map(len,by_rank));donor=next((indices[0] for indices in by_rank if indices),None)
    waves=[]
    for first in range(0,slots,micro):
        waves.append([[(indices[i],True) if i<len(indices) else (donor,False)
                       for i in range(first,first+micro)] for indices in by_rank])
    return dict(rows=rows,owners=owners,loads=loads,waves=waves,global_episode_denominator=len(episodes))


def group_rewards(draw,generated,snapshot,learning,*,labels_for,score_episode,reward,advantages):
    """Policy-independent CPU preparation after real responses, before reference forward.

    Feed original scorer call provenance and its own registered group advantage
    function. The full consumer reconciles these rows before optimizer commit.
    """
    gid=draw['group_id']
    require(gid==generated['group_id'] and draw['query_draw_id']==generated['query_draw_id'] and
            snapshot['rollout_scope']=='node','wrong group or unsupported rollout scope')
    k=learning['kernel'];es=generated['episodes']
    require(len(es)==k['G'] and [e['replicate'] for e in es]==list(range(k['G'])),'incomplete reward group')
    output=[];requests=set(); eids=set()
    for ep in es:
        eid=ep['episode_id']
        require(eid not in eids and not ep['synthetic'],'reused or synthetic reward episode');eids.add(eid)
        offline={}
        for i,call in enumerate(ep['calls']):
            packet=call['packet'];node=packet['stage'];rid=packet['request_id']
            require(node in snapshot['target_nodes'] and rid not in requests and
                    packet['fresh_capture_identity']['episode_id']==eid,'wrong reward call')
            requests.add(rid)
            offline.setdefault(node,dict(calls=[]))['calls'].append(dict(online=call['online'],output=call['output'],
                labels=labels_for(node,call['online'],call['annotation']),call_ids=[eid+':c'+str(i)]))
        components=score_episode(dict(offline=offline),learning['scorer'])
        output.append(dict(group_id=gid,episode_id=eid,components=components,reward=reward(components,snapshot)))
    vals,_,_=advantages([row['reward'] for row in output],k['algorithm'],k['epsilon'])
    return [dict(row,advantage=v) for row,v in zip(output,vals,strict=True)]
