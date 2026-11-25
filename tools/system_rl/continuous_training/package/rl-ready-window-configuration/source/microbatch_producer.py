"""Produce scheduled, globally padded actor inputs from closed recording groups.

The caller supplies pin-checked generation and GROUP readers. No joined cohort
report or previously packed tensors are inputs. Source objects are private to
one immutable policy window. The normal full-cohort admission still owns Adam.
"""
from functools import lru_cache
import hashlib
from dependencies import require
from microbatch_fields import micro_fields
from microbatch_roster import roster


@lru_cache(maxsize=2)
def mapping(source, expected):
    raw=source.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'changed mapping source')
    body=raw.decode().split('def build(files, config, session=None):')
    require(len(body)==2,'unknown consumer mapping interface')
    start="        prov = provenance[draw]"; end="    import numpy as np"
    block=body[1]
    require(block.count(start)==block.count(end)==1,'unknown consumer mapping boundary')
    from textwrap import dedent,indent
    block=dedent(block[block.index(start):block.index(end)]).splitlines()[1:]
    # The actor row is global even when preceding groups have not been packed.
    src='\n'.join(block)
    old="m['actor_row'] = len(selected)"
    require(src.count(old)==1,'unknown actor row enumeration')
    src=src.replace(old,"m['actor_row'] = ordinals[g['group_id'],e['episode_id'],c['request_identity']['request_id']]")
    code='def rows(draw,prov,g,snap,advantages,ordinals):\n    selected=[];all_mapping=[]\n'+indent(src,'    ')+'\n    return selected\n'
    ns={};exec(compile(code,str(source)+':row-mapping','exec'),ns)
    return ns['rows']


class Producer:
    def __init__(self, plan, generated, snapshot, *, world, micro, accumulation, group_size,
                 source, expected, pad, context_limit, reward_rows, origins, group_reader):
        self.plan=plan; self.snapshot=snapshot; self.origins=origins;self.read=group_reader
        self.groups={};self.draws={};self.advantages={};maximum=[0,0]
        self.kw=dict(source=source,consumer_sha256=expected,snapshot=snapshot,pad=pad,context_limit=context_limit)
        self.mapping=mapping(source,expected)
        def records():
            for d,g in zip(plan['draws'],generated,strict=True):
                gid=d['group_id'];self.draws[gid]=d
                require(g['policy']==plan['policy'],'different rollout policy')
                for r in reward_rows(d,g):
                    k=(gid,r['episode_id'])
                    require(r['group_id']==gid and k not in self.advantages,'duplicate reward')
                    self.advantages[k]=r['advantage']
                for ep in g['episodes']:
                    require((gid,ep['episode_id']) in self.advantages,'missing reward')
                    for call in ep['calls']:
                        if snapshot['trainable_mask'][call['packet']['stage']] and call['token_ids']:
                            maximum[0]=max(maximum[0],len(call['prompt_token_ids']))
                            maximum[1]=max(maximum[1],len(call['token_ids']))
                yield g
        self.layout=roster(plan['draws'],records(),world=world,micro=micro,accumulation=accumulation,
            trainable_mask=snapshot['trainable_mask'],group_size=group_size,policy=plan['policy'])
        require(list(self.draws)==plan['group_ids'] and len(self.advantages)==self.layout['global_episode_denominator'],
                'different selected group or episode roster')
        self.keys=[tuple(r[:3]) for r in self.layout['rows']]
        self.ids={key:i for i,key in enumerate(self.keys)}
        self.geometry=tuple(maximum)

    def group(self,gid):
        if gid not in self.groups:
            d=self.draws[gid]; g=self.read(d)
            require(g['policy']==self.plan['policy'] and g['synthetic'] is False and
                    g['group_id']==gid and g['query_draw_id']==d['query_draw_id'],'different closed group')
            b=g['online']['behavior_bundle']
            require(b['online_gold_injected'] is False and b['synthetic'] is False and
                    b['global_query_count']==len(b['groups'])==1,'invalid recording group')
            one,=b['groups'];r=d['record'];o=self.origins[r['recording_ref_id']]
            require(one['snapshot']==self.snapshot and one['group_id']==gid and one['query_id']==r['sample_id'] and
                    one['query_draw_id']==d['query_draw_id'] and one['recording_id']==o['capture_recording_id'] and
                    one['closed'] and one['fresh_on_policy'],'different policy snapshot or recording')
            prov={**r,'capture_recording_id':o['capture_recording_id'],'recording_filename':o['recording_filename']}
            rows=self.mapping(d['draw_index'],prov,one,self.snapshot,self.advantages,self.ids)
            local={m['actor_row']:(c,m,adv) for c,m,adv in rows}
            require(len(rows)==len(local) and set(local)=={i for i,k in enumerate(self.keys) if k[0]==gid},
                    'different trainable rows after group closure')
            self.groups[gid]=local
        return self.groups[gid]

    def wave(self,schedule):
        ranks=[]
        for slots in schedule:
            rows=[self.group(self.keys[i][0])[i] for i,owned in slots]
            tensors,non=micro_fields(rows,self.geometry,**self.kw)
            ranks.append((slots,tensors,non))
        return ranks


def rank_transport(producer,rank,meta,*,agree_ready):
    """The original actor pulls only this rank's microbatches, one ahead.

    Returns the deferred input plus its pre-Adam gate. The caller supplies
    actual globally accepted metadata and a collective error rendezvous.
    """
    from microbatch_fields import CommitGate,scheduled_transport
    layout=producer.layout
    require(meta['global_episode_denominator']==layout['global_episode_denominator'] and
            meta['snapshot']==producer.snapshot and not any(producer.snapshot['auxiliary_coefficients'].values()),
            'only the admitted policy-only scope can use provisional inputs')
    waves=[w[rank] for w in layout['waves']]
    gate=CommitGate(waves,meta)
    def prepare(slots):
        _,t,n=producer.wave([slots])[0];gate.observe(slots,t,n)
        return t,n
    # These are the key sets inspected before verl calls select(); the actual
    # values are produced only for the requested rows in that microbatch.
    data=scheduled_transport(waves,prepare,meta,
        ('responses','response_mask','input_ids','attention_mask','position_ids','old_log_probs','advantages','prompts','ref_log_prob'),
        ('multi_modal_inputs','uid','record_identity','adapter_route'),micro=len(waves[0]),agree_ready=agree_ready)
    return data,gate
