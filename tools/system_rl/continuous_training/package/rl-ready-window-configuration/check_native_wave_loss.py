"""Compare native verl CPU updates for an eager packet and independently streamed rows.

Real GRPO fields, registered loss kernel and eight actual rank schedules; small
differentiable logprob head, independent CPU ranks (no Omni or FSDP claim).
"""
from types import SimpleNamespace
from unittest.mock import patch
import copy,json
from pathlib import Path
import torch

def compare(producer,packet,root):
    from dependencies import ROOT,load,require
    from run_package import pin
    from loss_kernel import VerlKernel
    from microbatch_producer import rank_transport
    from microbatch_fields import run_scheduled_update
    from check_accepted_actor_packet import equal
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    from verl import DataProto
    import numpy as np
    rt=json.loads((ROOT/'jev-system-rl/implementation/runtime.json').read_bytes())['training']
    bridge=load(ROOT/'unified-autograd-bridge/bridge.py','wave_test_bridge')
    kernel=VerlKernel(bridge.SourceKernel({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')}),
                      core_sha256=rt['core_sha256'],clip_ratio=packet['clip'])
    registry=get_policy_loss_fn.__globals__['POLICY_LOSS_REGISTRY'];tag='real_wave_cpu_test'
    require(tag not in registry,'registry collision')
    den=producer.layout['global_episode_denominator'];world=len(producer.layout['waves'][0])
    meta=dict(global_episode_denominator=den,snapshot=producer.snapshot,group_size=packet['learning']['kernel']['G'],global_query_count=len(producer.draws))
    beta=packet['learning']['kernel']['beta']
    require(beta==packet['meta']['beta'],'different KL coefficient')
    totals=[]
    class Actor:
        update_policy=DataParallelPPOActor.update_policy
        def __init__(self,micro,n,order,owned):
            self.actor_module=torch.nn.Linear(1,1,bias=False)
            with torch.no_grad():self.actor_module.weight.fill_(.25)
            self.actor_optimizer=torch.optim.Adam(self.actor_module.parameters(),lr=.001,foreach=False,fused=False)
            self.config=SimpleNamespace(use_kl_loss=False,use_dynamic_bsz=False,ppo_epochs=1,ppo_mini_batch_size=n,ppo_micro_batch_size_per_gpu=micro,
                entropy_coeff=0.,loss_agg_mode='token-mean',calculate_entropy=False,use_rollout_log_probs=True,policy_loss={'loss_mode':tag})
            self.scaler=None;self.use_prefix_grouper=False;self.cursor=0;self.losses=[];self.order=order;self.owned=owned
        def _forward_micro_batch(self,inputs,**kw):
            # Same differentiable CPU input in both arms, not an Omni logprob observation.
            old=inputs['old_log_probs']; current=old-.01*self.actor_module.weight.square()
            return dict(log_probs=current)
        def _optimizer_step(self):
            self.grad=self.actor_module.weight.grad.detach().clone()
            self.actor_optimizer.step()
            return self.grad.norm()
        def loss(self,log_prob,old_log_prob,advantages,response_mask,**kw):
            vals=[]
            for i,x in enumerate(log_prob.split(1)):
                j=self.cursor+i;row=self.order[j];t=packet['tensors'];k='ref_log_prob';sl=slice(row,row+1)
                loss,_=kernel.loss(x,old_log_prob[i:i+1],t[k][sl],advantages[i:i+1],response_mask[i:i+1],
                    t['unified_policy_node_weight'][sl],t['unified_kl_node_weight'][sl],den,beta)
                vals.append(loss*int(self.owned[j]))
            self.cursor+=len(vals);v=torch.stack(vals).sum();self.losses.append(v.detach().clone())
            return v*world*(len(self.order)//len(vals)),{}
    try:
        with patch('verl.workers.actor.dp_actor.get_device_id',return_value='cpu'):
            for rank in range(world):
                waves=[w[rank] for w in producer.layout['waves']];micro=len(waves[0]);flat=sum(waves,[])
                order=[i for i,_ in flat];owned=[o for _,o in flat]
                events=json.loads((root.parent/f'rank-{rank}'/'LOSS.json').read_bytes())['result']['events']
                actual=[(e['row'],e['owned']) for e in events];require(actual==flat,'different actual schedule')
                actual_waves=[actual[j:j+micro] for j in range(0,len(actual),micro)]
                saved=None
                for stream in (False,True):
                    a=Actor(micro,len(order),order,owned);registry[tag]=a.loss
                    if stream:
                        data,gate=rank_transport(producer,rank,meta,agree_ready=lambda fn:fn())
                        run_scheduled_update(a,data,reconcile=lambda:gate.reconcile(packet,actual_waves),agree_ready=lambda fn:fn())
                        require(gate.accepted,'missing input gate')
                    else:
                        tensors={k:v[order] for k,v in packet['tensors'].items()}
                        non={}
                        for k,v in packet['non_tensors'].items():
                            arr=np.empty(len(order),dtype=object);arr[:]=[v[i] for i in order];non[k]=arr
                        data=DataProto.from_dict(tensors=tensors,non_tensors=non,meta_info={'temperature':1.})
                        a.update_policy(data)
                    require(a.cursor==len(order) and a.grad.isfinite().all(),'incomplete native backward')
                    state=[a.losses,a.grad,a.actor_module.state_dict(),a.actor_optimizer.state_dict()]
                    if stream:equal(state,saved)
                    else:saved=copy.deepcopy(state)
                totals.append(float(a.grad.norm()))
    finally:registry.pop(tag,None)
    require(any(t>0 for t in totals),'no gradient signal in replay')
    return dict(ranks=len(totals),device='cpu',head='scalar',actual_grpo_kernel=True,loss_gradient_adam_max_abs=0.,
                fsdp_validated=False)
