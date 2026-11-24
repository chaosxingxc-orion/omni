"""A single full actor update in isolation on the sealed previous-profile OOM cohort."""
import time


def measure(spec,out,rank,world,cfg,sealed):
    import hashlib
    from pathlib import Path
    import torch
    import torch.distributed as dist
    from bounded_actor_step import actor_batch_geometry,compact_scores,prepare
    from worker_dp8 import construct
    from events import run_existing_actor,partition
    from storage import commit
    from lora_row_precision import attach_hf
    from dependencies import require
    from run_package import read
    q=spec['profile_batch_check'];profile=read(sealed(q['profile']))
    cfg=dict(cfg,expanded_publication='/sealed/initial-adapter')
    worker,adapted,_=construct(cfg,out)
    packet=torch.load(sealed(q['packet']),weights_only=True,mmap=True,map_location='cpu')
    old=read(sealed(q['packet_receipt']))
    require(packet['dataset']==old['dataset'] and old['payload']['sha256']==spec['files'][q['packet']]
            and packet['report']['actor_rows']==q['rows'], 'different original OOM batch')
    _,fields,plan,runtime,_=prepare(dict(spec,dataset_sha256=packet['dataset']['sha256']),sealed,world,packet)
    batch=profile['actor_batch'];micro=batch['microbatch_per_card'];acc=batch['gradient_accumulation_steps']
    layout=partition(plan,fields,world,micro,acc);events=layout['schedules'][rank]
    review=q.get('natural_action_review')
    if review:
        require(q.get('eval_only') is True and not q.get('trace_rows'),'same-action review is isolated')
        sealed('initial_action_review.py')
        from initial_action_review import action_fields
        actions=read(sealed(review))
        require(actions['dataset']==packet['dataset'] and
                actions['policy_version']=='initial-sha256:'+hashlib.sha256(
                    Path('/sealed/initial-adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
                'different evaluation population or weights')
        action_fields(fields,actions['rows'])
    eval_width=q.get('eval_width',micro)
    if type(eval_width) is not int or eval_width<1 or micro%eval_width:
        raise ValueError('evaluation width must divide the training microbatch')
    close=attach_hf(adapted.get_base_model().model,spec['lora_row_geometry'])
    module=worker.actor.actor_module
    device=next(module.parameters()).device
    def local_input(rows):
        inputs={k:v[rows].to(device) for k,v in fields['tensors'].items()}
        inputs.update({k:v[rows] for k,v in fields['non_tensors'].items()})
        inputs['multi_modal_inputs']=[{k:v.to(device) for k,v in mm.items()} for mm in inputs['multi_modal_inputs']]
        return inputs
    torch.use_deterministic_algorithms(True)
    trace_rows=q.get('trace_rows')
    if trace_rows is not None:
        require(q.get('eval_only') is True and eval_width==1 and trace_rows and
                all(type(row) is int and 0<=row<len(layout['owners']) for row in trace_rows),
                'predictor trace requires independent no-update rows')
        # All FSDP ranks participate in each actual forward. Only rank zero writes.
        events=[dict(row=row,owned=True) for row in trace_rows]
    with actor_batch_geometry(adapted,batch['arithmetic']):
        module.eval();scores=[]
        with torch.no_grad():
            for start in range(0,len(events),eval_width):
                rows=[e['row'] for e in events[start:start+eval_width]]
                inputs=local_input(rows)
                from contextlib import ExitStack
                with ExitStack() as stack:
                    if trace_rows is not None and rank==0:
                        from initial_predictor_trace import observe_hf
                        sealed('initial_predictor_trace.py')
                        stack.enter_context(observe_hf(adapted,inputs,out/'traces'/str(rows[0])))
                    actual=compact_scores(worker.actor,inputs).cpu()
                for i,row in enumerate(rows):
                    if events[start+i]['owned']:
                        mask=fields['tensors']['response_mask'][row].bool()
                        old_scores=(actual.new_tensor(actions['rows'][row]['log_probs']) if review
                                    else fields['tensors']['old_log_probs'][row][mask])
                        values=actual[i][mask]
                        scores.append(dict(row=row,log_probs=values.tolist(),source_max_abs=float((old_scores-values).abs().max())))
        commit(out/f'rank-{rank}/PROFILE-EVAL.json',dict(dataset=packet['dataset'],rows=scores,
               **({'action_review_sha256':spec['files'][review]} if review else {})))
        if q.get('eval_only') is True:
            close();worker.close_actor_arithmetic()
            commit(out/f'rank-{rank}/RESULT.json',dict(eval_width=eval_width,formal_updates=0,
                   owned_rows=[r['row'] for r in scores],isolated_update=False))
            return
        module.train();dist.barrier();torch.npu.synchronize();torch.npu.reset_peak_memory_stats();t=time.monotonic()
        result=run_existing_actor(worker.actor,fields,plan,None,lambda inputs:compact_scores(worker.actor,inputs),
            actor_sha256=runtime['actor_sha256'],forward_api=runtime['forward_api'],world=world,rank=rank,
            micro_batch=micro,accumulation=acc,lora_dropout=cfg['lora_dropout'])
        torch.npu.synchronize()
        report=dict(seconds=time.monotonic()-t,peak_allocated=torch.npu.max_memory_allocated(),
                    peak_reserved=torch.npu.max_memory_reserved(),microbatches=len(events)//micro,
                    owned_rows=result['local_owned_rows'],objective=result['objective_observation'],
                    isolated_update=bool(result['optimizer_step']),formal_updates=0)
    close();worker.close_actor_arithmetic()
    commit(out/f'rank-{rank}/RESULT.json',report)
