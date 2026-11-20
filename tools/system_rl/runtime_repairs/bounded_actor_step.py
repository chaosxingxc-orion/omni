"""Run one accepted cohort through the original actor, schedule and loss."""
import hashlib
from pathlib import Path


def compact_scores(actor,inputs):
    import torch
    from forward import forward
    from dependencies import require
    mask=inputs['attention_mask'][0].bool(); actions=inputs['response_mask'][0].bool()
    width=len(actions); length=int(actions.sum()); plen=len(mask)-width
    require(len(inputs['input_ids'])==1 and 0<length<=width and torch.equal(mask[-width:],actions)
            and bool(mask[plen-1]) and torch.equal(actions,actions.long().cumprod(0).bool())
            and bool((mask[:plen].long().diff()>=0).all()),'expected left-padded prompt / right-padded actions')
    one=dict(inputs)
    for key in ('input_ids','attention_mask','position_ids'):one[key]=inputs[key][...,mask]
    for key in ('responses','response_mask'):one[key]=inputs[key][...,:length]
    scores=forward(actor,one,temperature=1.,feature_dtype=torch.bfloat16)[1]
    return torch.nn.functional.pad(scores,(0,width-length))


def run(worker,adapted,spec,out,sealed,rank,world):
    import torch
    import torch.distributed as dist
    import numpy as np
    from dependencies import ROOT,require,upstream
    from run_package import pin
    from storage import commit,check_running
    from loss_kernel import VerlKernel
    sealed('events.py')
    from events import run_existing_actor
    import inspect
    require(Path(inspect.getfile(run_existing_actor)).resolve()==sealed('events.py').resolve(),'unexpected actor event implementation')
    from runtime_builder import read
    # Consumer validation happened on CPU, pin-check before deserializing its local handoff.
    packet=torch.load(sealed(spec['accepted_fields']),weights_only=False,map_location='cpu')
    fields=dict(tensors=packet['tensors'],non_tensors={},meta_info=packet['meta'])
    for k,values in packet['non_tensors'].items():
        a=np.empty(len(values),dtype=object);a[:]=values;fields['non_tensors'][k]=a
    report=packet['report'];task=upstream()
    runtime=read(ROOT/'jev-system-rl/implementation/runtime.json')['training']
    require(not packet['meta']['synthetic'] and report['required_fields_validated'] and not packet['clock']['successful_optimizer_steps']
            and not any(g['consumed'] for g in packet['clock']['groups'].values()) and spec['dataset_sha256']==packet['dataset']['sha256'],
            'expected reserved initial TRAIN batch')
    require(spec['image']==runtime['image_id'] and spec['training']['actor_arithmetic_profile']=='omni-bf16-tp4-fsdp4-v1','unverified actor image/profile')
    kernel=task.bridge.SourceKernel({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    base=task.bridge.PreparedBatch(fields,report,VerlKernel(kernel,core_sha256=runtime['core_sha256'],clip_ratio=packet['clip']))
    online={tuple(item['identity'][n] for n in ('group_id','episode_id','call_id','node')):item['online_input'] for item in packet['online']}
    plan=task.prepare(base,packet['sidecar'],online)
    require(not any(plan.coefficients.values()),'bounded policy-only step requires zero auxiliary coefficients')
    heard=[]; worker.actor.actor_module.train()
    def evaluate(inputs):
        scores=compact_scores(worker.actor,inputs); mask=inputs['response_mask'].bool()
        row=list(fields['non_tensors']['record_identity']).index(inputs['record_identity'][0])
        old=fields['tensors']['old_log_probs'][row:row+1].to(scores)
        delta=(scores.detach()-old)[mask];m=float(delta.abs().max());seq=float(delta.sum().abs())
        require(m<=.01 and seq<=.01,'effective train/behavior parity failed')
        heard.append(dict(row=row,tokens=int(mask.sum()),max_abs=m,sequence_abs=seq))
        return scores
    def before_step(_):
        check_running(out,spec['storage_budget_bytes'])
        commit(out/f'rank-{rank}/BACKWARDS.json',dict(actions=heard,rank=rank))
    result=run_existing_actor(worker.actor,fields,plan,None,evaluate,actor_sha256=runtime['actor_sha256'],
        world=world,rank=rank,before_step=before_step,forward_api=runtime['forward_api'])
    require(result['native_optimizer_step'],'real optimizer did not update')
    audit=worker.audit_master_optimizer(1)
    worker.actor_lr_scheduler.step()
    commit(out/f'rank-{rank}/LOSS.json',dict(rank=rank,audit=audit,result=result,actions=heard,step=1))
    from policy_identity import artifact,describe
    worker.export_adapter(out/'step-1')
    # Every rank independently reads the published BF16 delta.
    public=artifact(out/'step-1/lora_adapter/adapter_model.safetensors')
    require(public['nonzero_effective_delta'],'exported empty effective update')
    commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,step=1,actions=heard,objective=result['objective_observation'],
            metrics=result['metrics'],publication_sha256=public['identity_sha256'],adapter_export=True,arithmetic=worker.close_actor_arithmetic()))
