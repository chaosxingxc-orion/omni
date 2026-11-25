"""Fixed-image CPU check of native verl update_policy and seeded Adam state.

Uses the actual method with a tiny differentiable CPU head, not the Omni model.
Injects a producer failure after the first completed backward.
"""
import copy,json
from unittest.mock import patch
from threading import Event
from types import SimpleNamespace
import torch,numpy as np
from microbatch_fields import scheduled_transport,run_scheduled_update

def check():
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    from verl import DataProto
    from check_accepted_actor_packet import equal
    from dependencies import require
    torch.manual_seed(17)
    class Actor:
        update_policy=DataParallelPPOActor.update_policy
        def __init__(self):
            self.actor_module=torch.nn.Linear(1,1,bias=False)
            self.actor_optimizer=torch.optim.Adam(self.actor_module.parameters(),lr=.01,foreach=False,fused=False)
            self.actor_module(torch.ones(1,1)).sum().backward();self.actor_optimizer.step();self.actor_optimizer.zero_grad()
            self.config=SimpleNamespace(use_kl_loss=False,use_dynamic_bsz=False,ppo_epochs=1,ppo_mini_batch_size=4,ppo_micro_batch_size_per_gpu=2,
                entropy_coeff=0.,loss_agg_mode='token-mean',calculate_entropy=False,use_rollout_log_probs=True,policy_loss={'loss_mode':'microbatch_cpu_test'})
            self.scaler=None;self.use_prefix_grouper=False;self.steps=0;self.backward_seen=Event()
            self.actor_module.weight.register_hook(lambda g:self.backward_seen.set())
        def _forward_micro_batch(self,inputs,**kw):return dict(log_probs=self.actor_module(inputs['input_ids'].float()))
        def _optimizer_step(self):
            self.steps+=1;self.actor_optimizer.step()
            return torch.zeros(())
    def loss(log_prob,advantages,**kw):return (log_prob*advantages).mean(),{}
    registry=get_policy_loss_fn.__globals__['POLICY_LOSS_REGISTRY'];assert 'microbatch_cpu_test' not in registry
    registry['microbatch_cpu_test']=loss
    def input_for(slots):
        x=torch.tensor(slots)[:,None]+1
        t=dict(responses=x,response_mask=x.ne(0),input_ids=x,attention_mask=x.ne(0),position_ids=x-1,old_log_probs=x.float().neg(),advantages=x.float())
        non={k:np.array(slots,dtype=object) for k in ('record_identity','adapter_route')}
        return t,non
    start=Actor();weights=copy.deepcopy(start.actor_module.state_dict());optim=copy.deepcopy(start.actor_optimizer.state_dict())
    def actor():
        a=Actor();a.actor_module.load_state_dict(copy.deepcopy(weights));a.actor_optimizer.load_state_dict(copy.deepcopy(optim));return a
    waves=[[0,1],[2,3]];t,non=input_for([0,1,2,3]);baseline=actor()
    try:
        baseline.update_policy(DataProto.from_dict(tensors=t,non_tensors=non,meta_info={'temperature':1.}))
        for failure in (None,"wave","admission"):
            fail=failure is not None
            a=actor()
            def prep(slots):
                if failure=='wave' and slots==waves[1]:
                    require(a.backward_seen.wait(15),'first backward did not execute');raise ValueError('injected second-wave failure')
                return input_for(slots)
            data=scheduled_transport(waves,prep,{},t.keys(),non.keys(),micro=2,agree_ready=lambda fn:fn())
            def reconcile():
                if failure=='admission':raise ValueError('injected admission failure')
            try:run_scheduled_update(a,data,reconcile=reconcile,agree_ready=lambda fn:fn())
            except ValueError as e:
                if not fail or str(e) not in ('injected second-wave failure','injected admission failure'):raise
            require(a.backward_seen.is_set() and a.steps==int(not fail),'wrong optimizer boundary')
            equal(a.actor_module.state_dict(),weights if fail else baseline.actor_module.state_dict())
            equal(a.actor_optimizer.state_dict(),optim if fail else baseline.actor_optimizer.state_dict())
            require(all(p.grad is None for p in a.actor_module.parameters()),'left partial gradients')
        return dict(native_update_policy=True,device='cpu',head='linear',success_matches_eager_adam=True,
                    second_wave_failed_after_backward=True,parameters_and_adam_unchanged=True,partial_gradients_cleared=True,final_admission_failure_prevents_adam=True)
    finally:del registry['microbatch_cpu_test']
if __name__=='__main__':
    with patch('verl.workers.actor.dp_actor.get_device_id',return_value='cpu'):
        print(json.dumps(check()))
