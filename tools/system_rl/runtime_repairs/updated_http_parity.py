"""Compare sampled HTTP actions against the actual recovered FSDP8 worker."""
import argparse
from contextlib import contextmanager
import hashlib
from itertools import accumulate, pairwise
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def prepare_replayed_top1(pairs, activation, *, limits):
    """Bind diagnostic argmax to matching natural production action evidence."""
    import math

    def observed(receipt):
        raw = receipt['raw'].encode()
        if receipt['status'] != 200 or sha(raw) != receipt['response_sha256']:
            fail('HTTP response bytes differ')
        body = json.loads(raw)
        choice, = body['choices']
        prompt = body['prompt_token_ids']
        tokens = choice['token_ids']
        items = choice['logprobs']['content']
        if (body['model'] != 'qwen3-omni-30b-a3b-instruct' or not prompt or not tokens
                or [x['token'] for x in items] != [f'token_id:{t}' for t in tokens]
                or any(not math.isfinite(x['logprob']) or x['logprob'] == -9999
                       or x['logprob'] > 0 for x in items)):
            fail('incomplete raw action evidence')
        return prompt, tokens, items

    def wire_hash(wire):
        return sha(json.dumps(wire, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=False).encode())

    rows = []
    for pair in pairs:
        packet = pair['packet']
        wire = pair['wire']
        replay_wire = pair['replay_wire']
        expected_replay = dict(wire, top_logprobs=2, max_tokens=len(replay_wire['vllm_xargs']['decode_action_ids']),
                               vllm_xargs=replay_wire['vllm_xargs'])
        expected_replay['cache_salt'] = replay_wire['cache_salt']
        if (wire['messages'] != packet['wire']['messages']
                or wire['top_logprobs'] != 0
                or replay_wire != expected_replay
                or set(replay_wire['vllm_xargs']) != {'decode_action_ids'}
                or wire['temperature'] != 1 or wire['top_p'] != 1
                or wire['chat_template_kwargs'] != {'enable_thinking': False}):
            fail('natural and replay wire semantics differ')
        for name, payload in [('natural', wire), ('replay', replay_wire)]:
            intent = pair[name + '_intent']
            if intent['wire_sha256'] != wire_hash(payload) or intent['automatic_retry']:
                fail('request wire custody differs')
        prompt, tokens, original = observed(pair['natural'])
        replay_prompt, replay_tokens, diagnostic = observed(pair['replay'])
        if (replay_prompt != prompt or replay_tokens != tokens
                or replay_wire['vllm_xargs']['decode_action_ids'] != tokens
                or [x['logprob'] for x in original] != [x['logprob'] for x in diagnostic]):
            fail('diagnostic actions or probabilities differ from production')
        argmax = []
        for item in diagnostic:
            candidates = item['top_logprobs']
            if (len(candidates) < 2 or len({x['token'] for x in candidates}) != len(candidates)
                    or any(not x['token'].startswith('token_id:') or not math.isfinite(x['logprob'])
                           or x['logprob'] == -9999 for x in candidates)):
                fail('missing raw top candidates')
            ordered = sorted(candidates, key=lambda x: x['logprob'], reverse=True)
            if ordered[0]['logprob'] == ordered[1]['logprob']:
                fail('ambiguous top-1 tie requires raw engine ranks')
            argmax.append(int(ordered[0]['token'].removeprefix('token_id:')))
        rows.append(dict(lane=pair['lane'], packet=dict(packet, wire=wire, wire_sha256=wire_hash(wire)),
                         prompt=prompt, tokens=tokens, logprobs=[x['logprob'] for x in original],
                         raw_top1=argmax, adapter_sha=activation['weights_ref']['sha256'],
                         publication_sha=activation['publication_ref']['sha256'],
                         response=pair['natural']['response_sha256'],
                         elapsed=pair['natural']['elapsed_seconds'], source_refs=pair['source_refs']))
    if not rows or not (0 <= limits['token'] <= .01 and 0 <= limits['sequence'] <= .01
                        and limits['max_argmax_flips'] == 0):
        fail('invalid parity roster or limits')
    return dict(schema='production-with-replayed-top1-parity-v1', rows=rows,
                limits=limits, training_use=False)


def review_replayed_top1(sealed, activation, checkpoint_ref, exit_receipt, rank_receipts, *, limits):
    """Review the fixed FSDP8 same-action score against its admitted TP4 adapter."""
    import math

    if (sealed['schema'] != 'production-with-replayed-top1-parity-v1'
            or sealed['training_use'] is not False or sealed['limits'] != limits
            or not 0 <= limits['token'] <= .01 or not 0 <= limits['sequence'] <= .01
            or limits['max_argmax_flips'] != 0):
        fail('invalid replay comparator contract')
    rows = sealed['rows']
    if not 0 < len(rows) <= 8 or len({row['lane'] for row in rows}) != len(rows):
        fail('invalid replay comparator roster')
    if (exit_receipt['exit_code'] != 0 or exit_receipt['failure'] is not None
            or exit_receipt['complete_ranks'] != 8 or len(rank_receipts) != 8):
        fail('FSDP8 replay comparator did not complete')
    checkpoint_path = Path(checkpoint_ref['path'])
    checkpoint = read(checkpoint_path)
    if (file_sha(checkpoint_path) != checkpoint_ref['sha256']
            or checkpoint['world_size'] != 8
            or checkpoint['step'] != checkpoint['clock']['successful_optimizer_steps']):
        fail('FSDP8 replay comparator checkpoint differs')
    adapter_sha = activation['weights_ref']['sha256']
    adapter_member = 'lora_adapter/adapter_model.safetensors'
    if (activation['fixture'] is not False or activation['base'] is not False
            or activation['policy_version'] != 'checkpoint-sha256:' + adapter_sha
            or checkpoint['ranks'][0]['files'][adapter_member] != adapter_sha
            or sorted(worker['rank'] for worker in activation['workers']) != list(range(4))
            or any(worker['compared_original_keys'] != len(checkpoint['resident_identity']['publication']['tensors'])
                   for worker in activation['workers'])):
        fail('replay comparator adapter differs from checkpoint')
    maximum = {mode + '_' + key: 0.0 for mode in ('train', 'eval')
               for key in ('max_abs', 'sequence_abs')}
    for rank, score in enumerate(rank_receipts):
        row = rows[rank % len(rows)]
        if (score['rank'] != rank or score['restored_step'] != checkpoint['step']
                or score['checkpoint_sha'] != checkpoint_ref['sha256']
                or score['optimizer_updates'] != 0 or score['passed'] is not True
                or score['lane'] != row['lane'] or score['response_sha'] != row['response']
                or score['tokens'] != len(row['tokens']) or not row['tokens']
                or row['adapter_sha'] != adapter_sha or score['observed_policy_artifact'] != adapter_sha
                or row['publication_sha'] != activation['publication_ref']['sha256']):
            fail('FSDP8 comparator scored another action or policy')
        for mode in ('train', 'eval'):
            if score[mode + '_top1_flips']:
                fail('replay comparator top-1 changed')
            for suffix, limit in (('max_abs', limits['token']), ('sequence_abs', limits['sequence'])):
                key = mode + '_' + suffix
                value = score[key]
                if not math.isfinite(value) or not 0 <= value <= limit:
                    fail('replay comparator probability differs')
                maximum[key] = max(maximum[key], value)
    return dict(passed=True, step=checkpoint['step'], ranks=8, requests=len(rows),
                actions=sum(len(row['tokens']) for row in rows), top1_flips=0,
                checkpoint_sha256=checkpoint_ref['sha256'], adapter_sha256=adapter_sha,
                limits=limits, **maximum)


def prepare(root, wire_path, original, output=None, *, lanes=None, expected_tensors=None):
    """Reduce the original immutable HTTP/worker evidence on CPU."""
    import torch
    wire=read(wire_path)
    wire_sha=sha(json.dumps(wire,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
    candidates=[row for row in read(original)['rows'] if row['packet']['wire']['messages']==wire['messages']]
    example=next((row for row in candidates if row['packet']['wire'].get('seed')==wire.get('seed')),None)
    if example is None:raise ValueError('no original packet with matching prompt and seed')
    require=lambda value,reason: None if value else fail(reason)
    require(read(root/'activation-probe01/DRAINED.json')['guards_releasable'],'service not drained')
    rows=[]
    roster=[l['name'] for l in read(root/'activation-probe01/INTENT.json')['lanes']]
    if lanes is not None and roster!=lanes:fail('actual endpoint roster differs')
    for lane in roster:
        state=root/lane
        act=read(state/'ACTIVATION.json')
        cache=read(state/'CACHE-POLICY.json')
        precision=read(state/'PRECISION.json')
        response=read(root/'activation-probe01'/lane/'RESPONSE.json')
        require(response['status']==200 and sha(response['raw'].encode())==response['response_sha256'],'HTTP response bytes')
        observation=read(state/'full-logits'/('WIRE-'+wire_sha+'.json'))
        require(observation['response_sha256']==response['response_sha256'] and observation['wire_sha256']==wire_sha
                and observation['activation_ref']['sha256']==file_sha(state/'ACTIVATION.json'),'observation custody')
        after=read(state/'full-logits'/Path(observation['full_logits_ref']['path']).parent.name/'COMPLETE.json')
        capture=state/'full-logits'/Path(observation['full_logits_ref']['path']).parent.name
        require(file_sha(capture/'COMPLETE.json')==observation['full_logits_ref']['sha256'],'capture seal')
        body=json.loads(response['raw'])
        item,=body['choices']
        prompt=body['prompt_token_ids'];tokens=item['token_ids'];data=item['logprobs']['content']
        require(prompt==example.get('prompt_token_ids',example.get('prompt')) and tokens and [u['token'] for u in data]==[f'token_id:{i}' for i in tokens]
                and after['prompt_token_ids']==prompt and after['response_token_ids']==tokens,'actual prompt and action binding')
        expected_ranks=set(range(len(act['service_identity']['physical_devices'])))
        require(expected_ranks and {w['rank'] for w in after['workers']}==expected_ranks and
                {w['rank'] for w in act['workers']}==expected_ranks and
                all((expected_tensors is None or w['compared_original_keys']==expected_tensors) and w['compared_original_keys']==len(w['loaded_tensors'])>0 and w['sha256']==act['weights_ref']['sha256'] for w in act['workers']) and
                cache['reset_succeeded'] and cache['prefix_cache'] and cache['kv_cache'] and
                {w['rank'] for w in precision['workers']}==expected_ranks, 'device proof')
        vocabulary=after['config']['policy']['vocabulary']
        require(type(vocabulary) is int and vocabulary>0,'invalid vocabulary')
        all_scores=[]
        for w in sorted(after['workers'],key=lambda w:w['rank']):
            ref=w['requests'][0]['ref'];p=capture/'workers'/f"rank-{w['rank']}"/Path(ref['path']).name
            meta=read(p)
            payload=p.with_suffix('.bin').read_bytes()
            require(file_sha(p)==ref['sha256'] and sha(payload)==meta['sha256']==meta['tensor_ref']['sha256'] and
                    meta['shape']==[len(tokens),vocabulary] and meta['dtype']=='torch.bfloat16' and meta['full_vocabulary'] and
                    meta['response_token_ids']==tokens and meta['prompt_token_ids']==prompt and
                    meta['identity']==dict(after['identity'],engine_request_id=after['engine_request_id'],adapter_id=act['adapter_id']),
                    'full vocabulary token and policy custody')
            tensor=torch.frombuffer(bytearray(payload),dtype=torch.bfloat16).reshape(len(tokens),vocabulary)
            require(bool(tensor.isfinite().all()),'nonfinite logits')
            all_scores.append(tensor.argmax(-1).tolist())
        require(all(s==all_scores[0] for s in all_scores),'TP workers disagree on argmax')
        traces=[w.get('prefix_trace') for w in after['workers']]
        trace_mode=after['config'].get('prefix_trace')
        if isinstance(trace_mode,dict):
            # A partial cache hit has no actual local execution for its prefix.
            keys=['positions','decoder.input',*(f'{i}.{k}' for i in range(48) for k in ('in','post'))]
            ops=['q','k','v','attn','o']
            if trace_mode.get('row_operands'):ops+=['full_input','base','indices','lora_a','lora_b']
            keys+=[('op.' if i==0 else f'op{i}.')+k for i in trace_mode['layers'] for k in ops]
            if trace_mode.get('branch_operands'):
                keys += [f'op{i}.{branch}.{site}' for i in trace_mode['layers'] for branches,sites in
                         [(('q','k','v'),('base','a','b')),(('in','post'),('hidden','residual','sum'))]
                         for branch in branches for site in sites]
            for t in traces:
                first=t['first_computed']
                require(type(first) is int and 0<=first<len(prompt)==t['prompt_length'],'scheduler prompt extent')
                expected={f'{k}:{max(first,a)}:{min(len(prompt),b)}' for k in keys for a,b in trace_mode['windows']
                          if max(first,a)<min(len(prompt),b)}
                require(expected and set(t['rows'])==expected,'incomplete absolute window capture')
        elif trace_mode:
            require(all(t is not None and len(t)=={True:194,'layer0':204,'layer1':214,'layer1-q-operands':215,'q-operands':205,'o-operands':205}[after['config']['prefix_trace']]
                        for t in traces),'incomplete TP layer traces')
        else:
            require(not any(traces),'unexpected layer traces')
        rows.append(dict(lane=lane,packet=dict(example['packet'],wire=wire,wire_sha256=wire_sha),prompt=prompt,
            tokens=tokens,logprobs=[x['logprob'] for x in data],raw_top1=all_scores[0],adapter_sha=act['weights_ref']['sha256'],
            publication_sha=act['publication_ref']['sha256'],response=response['response_sha256'],elapsed=response['elapsed_seconds'],
            **(dict(prefix_traces={str(w['rank']):w['prefix_trace'] for w in after['workers']}) if all(traces) else {})))
    result=dict(rows=rows,training_use=False)
    if output is not None:
        with output.open('x') as f:json.dump(result,f,allow_nan=False)
    return result


def fail(reason):
    raise ValueError(reason)


def compare_prefix(earlier, traced, hf_root):
    from math import fsum
    rows=[]
    if read(hf_root/'EXIT.json')['exit_code']!=0:fail('HF trace incomplete')
    controls=[read(hf_root/f'rank-{i}/RESULT.json') for i in range(8)]
    for base, trace in zip(read(earlier)['rows'],read(traced)['rows'],strict=True):
        fields=('lane','prompt','tokens','logprobs','raw_top1','adapter_sha','publication_sha')
        if any(base[k]!=trace[k] for k in fields):fail('instrumented HTTP request did not reproduce')
        group=[r for r in controls if r['lane']==base['lane']]
        if len(group)!=4 or any(r['response_sha']!=base['response'] or r['export_sha']!=base['adapter_sha']
                                or r['optimizer_updates'] or r['trace_differences'] for r in group):
            fail('did not restore the same published policy and action evidence')
        h=group[0]['public']['boundaries']
        if not all(h==r['public']['boundaries'] for r in group):fail('HF ranks differ')
        for rank,boundaries in trace['prefix_traces'].items():
            keys=['decoder.input',*(f'{i}.{k}' for i in range(48) for k in ('in','post'))]
            layer0='op.q.prompt' in boundaries
            if layer0:keys[2:2]=['op.'+k for k in ('q','k','v','attn','o')]
            domain={}
            for scope in ('prompt','predictors'):
                expected=[f'{k}.{scope}' for k in keys]
                def hf(k):
                    site,scope=k.rsplit('.',1)
                    return h[site+'.'+rank+'.'+scope] if site in {'op.q','op.k','op.v','op.attn'} else h[k]
                if any(hf(k)['shape']!=boundaries[k]['shape'] for k in expected):fail('different trace shape')
                found=[k for k in expected if hf(k)!=boundaries[k]]
                domain[scope]={'first_observed_difference':next(iter(found),None),'differing_boundaries':len(found),'observed':len(expected)}
            rows.append(dict(lane=base['lane'],rank=int(rank),**domain))
        delta=[a-b for a,b in zip(group[0]['public']['scores'],trace['logprobs'],strict=True)]
        if max(map(abs,delta))!=group[0]['public_to_tp4']['max_abs'] or abs(fsum(delta))!=group[0]['public_to_tp4']['sequence_abs']:
            fail('instrumented output does not reproduce HF comparison')
    return dict(schema='updated-policy-prefix-boundary-comparison-v1',rows=rows,training_admission=False)


@contextmanager
def published_adapter_control(worker, export, resident_identity):
    """Overlay the sealed BF16 values on the same HF leaves, then restore master."""
    import re
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from safetensors.torch import load_file
    from dependencies import require
    from policy_identity import canonical, describe, resident_lora_weights

    public = load_file(str(export), device='cpu')
    master = resident_lora_weights(worker)
    require(describe(master) == resident_identity['master'], 'restored FP32 LoRA identity')
    require(describe(public) == resident_identity['publication'] and public.keys() == master.keys(),
            'sealed BF16 LoRA identity')
    require(all(t.dtype == torch.float32 and public[k].dtype == torch.bfloat16
                and torch.equal(t.bfloat16(), public[k]) for k, t in master.items()),
            'BF16 publication does not match the rounded master')
    leaves = {}
    for name, module in worker.actor.actor_module.named_modules():
        clean = name.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')
        if isinstance(module, FSDP) and re.search(r'\.lora_[AB]\.default$', clean):
            key = canonical(clean+'.weight')
            require(key not in leaves, 'duplicate FSDP leaf')
            leaves[key] = module
    require(leaves.keys() == public.keys(), 'FSDP leaf roster does not match adapter')

    def copy(values):
        for key, module in leaves.items():
            with FSDP.summon_full_params(module, recurse=False, writeback=True, rank0_only=False):
                weight = module.module.weight
                require(weight.dtype == torch.float32 and weight.shape == values[key].shape,
                        'unexpected LoRA leaf parameter layout')
                with torch.no_grad():
                    weight.copy_(values[key])
        rounded = resident_lora_weights(worker)
        require(all(torch.equal(rounded[k], values[k]) for k in rounded), 'FSDP LoRA overlay readback')

    copy(public)
    try:
        yield dict(tensors=len(public), elements=sum(v.numel() for v in public.values()),
                   master_differing_elements=sum(int((v != public[k]).sum()) for k, v in master.items()))
    finally:
        copy(master)


def evaluate(worker,adapted,spec,sealed,out,rank,world,checkpoint_ref,checkpoint):
    import torch
    from dependencies import require
    from storage import commit
    from retained_group_parity import hf_scores
    path=sealed(spec['updated_http_parity']);packet_file=read(path);rows=packet_file['rows']
    limits=packet_file.get('limits')
    if limits is None:
        # Old closed evidence pre-dates configured windows. Source defaults in the pinned criterion.
        limits=spec['parity_limits']
    export=checkpoint['ranks'][0]['files']['lora_adapter/adapter_model.safetensors']
    require(world==checkpoint['world_size'] and 1<=len(rows)<=world and all(row['adapter_sha']==export for row in rows),'same restored export required')
    # All ranks perform the same number of real collectives; duplicate calls on
    # the HF side are controls, never counted as additional behavior actions.
    inputs=[];outputs=[]
    for i in range(world):
        r=rows[i%len(rows)];packet=r['packet']
        inputs.append(dict(group=i%len(rows),replicate=0,packet=packet,prompt_token_ids=r['prompt'],source_sha256=file_sha(path)))
        outputs.append(dict(group=i%len(rows),replicate=0,request_id=packet['request_id'],wire_sha256=packet['wire_sha256'],
            source_sha256=file_sha(path),seed=packet['wire']['seed'],token_ids=r['tokens'],behavior_logprobs=r['logprobs'],raw_top1=r['raw_top1']))
    batch=dict(schema='bounded-train-natural-requests-v1',rows=inputs)
    actions=dict(manifest_sha256=file_sha(path),records=outputs)
    worker.actor.actor_module.eval()
    if packet_file.get('schema')=='closed-rollout-probability-diagnostic-v1':
        from contextlib import ExitStack
        from retained_kv_candidate import attach
        from math import fsum
        require(packet_file['training_use'] is False and all(r['raw_top1'] is None for r in rows),
                'original rollout missing-top1 control only')
        def unwrap(m):
            while type(m).__name__ in ('CheckpointWrapper','FullyShardedDataParallel'):
                m=m.module if type(m).__name__=='FullyShardedDataParallel' else m._checkpoint_wrapped_module
            return m
        results=[]
        for control in [[], *packet_file['projection_controls']]:
            calls={}
            with ExitStack() as stack:
                if 'lora_row_geometry' in spec:
                    from lora_row_precision import attach_hf
                    finish=attach_hf(adapted.get_base_model().model,spec['lora_row_geometry'])
                    stack.callback(lambda: calls.update(lora_a=finish()))
                for entry in control:
                    name=entry['projection']
                    require(name in ('q_proj','o_proj') and name not in calls,'invalid projection control')
                    layers=entry['layers']
                    close=attach(adapted.get_base_model().model,layers,side='hf',unwrap=unwrap,
                                 training_vjp=True,projections=(name,))
                    calls[name]=None
                    stack.callback(lambda close=close,name=name: calls.update({name:close()}))
                observed,=hf_scores(worker,adapted,batch,actions,file_sha(path),rank,world,actor_forward=True)
            train_delta=[a-b for a,b in zip(observed['train_scores'],outputs[rank]['behavior_logprobs'],strict=True)]
            results.append(dict(projections=calls,eval=observed,train_max_abs=max(map(abs,train_delta)),
                                train_sequence_abs=abs(fsum(train_delta))))
        commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,lane=rows[rank%len(rows)]['lane'],
               response_sha=rows[rank%len(rows)]['response'],checkpoint_sha=checkpoint_ref['sha256'],
               optimizer_updates=0,training_admission=False,results=results,
               arithmetic=worker.close_actor_arithmetic()))
        return
    if spec.get('published_adapter_control'):
        target=outputs[rank]
        def forward_control(*, q_operands=False, family_shadow=False):
            result,=hf_scores(worker,adapted,batch,actions,file_sha(path),rank,world,
                              trace_group=target['group'],trace_layer0=spec.get('trace_layer0',False),
                              trace_layer1=spec.get('trace_layer1',False),
                              operand_layer=spec.get('operand_layer',0),
                              projection_family_shadow=family_shadow,
                              trace_q_operands=q_operands, q_replay_plan=spec.get('q_replay_plan') if q_operands else None)
            return result
        def differences(a,b):
            import math
            delta=[x-y for x,y in zip(a,b,strict=True)]
            return dict(max_abs=max(map(abs,delta)), sequence_abs=abs(math.fsum(delta)), delta=delta)
        original=forward_control()
        f=Path('/prior-checkpoint/lora_adapter/adapter_model.safetensors')
        require(file_sha(f)==export,'BF16 control artifact differs from observed service')
        with published_adapter_control(worker,f,checkpoint['resident_identity']) as weight_control:
            rounded=forward_control(q_operands=spec.get('trace_q_operands',False),
                                    family_shadow=spec.get('projection_family_shadow',False))
        worker.audit_master_optimizer(checkpoint['step'])
        commit(out/f'rank-{rank}/RESULT.json',dict(
            rank=rank,lane=rows[rank%len(rows)]['lane'],restored_step=checkpoint['step'],
            checkpoint_sha=checkpoint_ref['sha256'], export_sha=export, response_sha=rows[rank%len(rows)]['response'],
            optimizer_updates=0, weight_control=weight_control,
            master_to_public=differences(original['scores'],rounded['scores']),
            master_to_tp4=differences(original['scores'],target['behavior_logprobs']),
            public_to_tp4=differences(rounded['scores'],target['behavior_logprobs']),
            trace_differences=[k for k,v in original['boundaries'].items() if v!=rounded['boundaries'][k]],
            projection_family_shadow=rounded['boundaries'].get('projection_family_shadow'),
            master=original,public=rounded,arithmetic=worker.close_actor_arithmetic()))
        return
    row_receipt={}
    if 'lora_row_geometry' in spec:
        from check_lora_row_vjp import check
        from lora_row_precision import attach_hf,grouped_linear
        close=attach_hf(adapted.get_base_model().model,spec['lora_row_geometry'])
        try:
            scores,=hf_scores(worker,adapted,batch,actions,file_sha(path),rank,world,actor_forward=True)
        except BaseException:
            close(check_coverage=False)
            raise
        row_receipt=close()
        site=spec['lora_row_geometry']['site']
        widths={'qkvo_lora_a':(2048,4096),'qkv_lora_a':(2048,),'o_lora_a':(4096,)}[site]
        row_receipt['gradient_control']={str(w):check(grouped_linear,'npu',w) for w in widths}
    else:
        scores,=hf_scores(worker,adapted,batch,actions,file_sha(path),rank,world,actor_forward=True)
    errors=[a-b for a,b in zip(scores['train_scores'],outputs[rank]['behavior_logprobs'],strict=True)]
    replay_top=outputs[rank]['raw_top1']
    import math
    mt=max(map(abs,errors)); seq=abs(math.fsum(errors))
    flips=[i+1 for i,(a,b) in enumerate(zip(scores['train_top1'],replay_top,strict=True)) if a!=b]
    passed=math.isfinite(mt) and mt<=limits['token'] and seq<=limits['sequence'] and len(flips)<=limits['max_argmax_flips'] and scores['max_abs']<=limits['token'] and scores['sequence_abs']<=limits['sequence'] and len(scores['top1_flips'])<=limits['max_argmax_flips']
    commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,lane=rows[rank%len(rows)]['lane'],tokens=scores['tokens'],
        **(dict(lora_row=row_receipt) if row_receipt else {}),passed=passed,train_max_abs=mt,train_sequence_abs=seq,train_top1_flips=flips,eval_max_abs=scores['max_abs'],
        eval_sequence_abs=scores['sequence_abs'],eval_top1_flips=scores['top1_flips'],restored_step=checkpoint['step'],
        observed_policy_artifact=export,checkpoint_sha=checkpoint_ref['sha256'],response_sha=rows[rank%len(rows)]['response'],
        optimizer_updates=0,arithmetic=worker.close_actor_arithmetic()))


def compare_q_operands(earlier, traced, hf_root, site='q'):
    """First compare live inputs/weights; then locate the earliest divergent stage."""
    prefix = compare_prefix(earlier, traced, hf_root)
    rows = []
    for lane in read(traced)['rows']:
        matches = [r for r in prefix['rows'] if r['lane'] == lane['lane']]
        candidates = [read(hf_root/f'rank-{i}/RESULT.json') for i in range(8)]
        hf = next(r['public']['boundaries'][site+'_operands'] for r in candidates if r['lane'] == lane['lane'])
        for item in matches:
            rank = str(item['rank'])
            tp = lane['prefix_traces'][rank][site+'_operands']
            if hf['block'] != tp['block'] or hf['scale'] != tp['scale']:
                fail('inconsistent Q observation or scaling contract')
            bad_weights = [k for k,v in hf['weights'].items() if tp['weights'][k] != v]
            differences = {}
            for key in ('input', 'base', 'a', 'b'):
                indexes = [i for i,(a,b) in enumerate(zip(hf['traces'][key],tp['traces'][key],strict=True)) if a!=b]
                differences[key] = indexes
            first = ('operand' if bad_weights or differences['input'] else
                     next((k for k in ('base','a','b') if differences[k]), None))
            if first is None and item['prompt']['first_observed_difference'] == f'op.{site}.prompt':
                first = 'scale_add'
            block = differences.get(first, [])
            rows.append(dict(lane=lane['lane'],rank=item['rank'],different_weights=bad_weights,
                first_observed_difference=first,different_blocks=differences,
                first_different_position_window=None if not block else
                    [block[0]*hf['block'],min(len(lane['prompt']),(block[0]+1)*hf['block'])],
                hf_calls=hf['calls'],
                tp_prefill_calls=[c for c in tp['calls'] if c['prompt_rows']]))
    return dict(schema=f'updated-policy-{site}-operand-comparison-v1',rows=rows,training_admission=False)


def compare_o_lora_b(traced, hf_root):
    """Separate a changed B input from B's same-operand shape sensitivity."""
    if read(hf_root/'EXIT.json')['exit_code'] != 0:
        fail('incomplete HF O replay')
    vl = {row['lane']:row for row in read(traced)['rows']}
    rows = []
    for rank in range(8):
        obs = read(hf_root/f'rank-{rank}/RESULT.json')
        row = vl[obs['lane']]
        if obs['optimizer_updates'] or obs['trace_differences'] or obs['export_sha'] != row['adapter_sha']:
            fail('HF O replay policy differs')
        h = obs['public']['boundaries']['o_operands']
        for tp_rank, trace in row['prefix_traces'].items():
            t = trace['o_operands']
            if h['weights'] != t['weights'] or h['traces']['input'] != t['traces']['input'] or h['block'] != t['block'] or h['scale'] != t['scale']:
                fail('O projection input, weights, or scale differs')
            def changed(stage):
                return [i for i,(a,b) in enumerate(zip(h['traces'][stage],t['traces'][stage],strict=True)) if a!=b]
            a,b=changed('a'),changed('b')
            shapes=[c['prompt_rows'] for c in t['calls'] if c['prompt_rows']]
            checks=h['replay']['lora_b_chunks']
            if [tuple(c['window']) for c in checks] != list(pairwise(accumulate(shapes,initial=0))):
                fail('B replay does not match live scheduler shapes')
            rows.append(dict(hf_rank=rank,tp_rank=int(tp_rank),lane=obs['lane'],
                a_blocks=a,b_blocks=b,b_diff_outside_a=sorted(set(b)-set(a)),
                b_same_operand_differences=[{k:c[k]['different_elements'] for k in ('native_2d','native_3d','grouped','grouped_full')} for c in checks]))
    return dict(schema='updated-policy-o-lora-b-replay-v1',rows=rows,training_admission=False)


def compare_projection_replay(original, traced, hf_root, *, layer, site):
    """Bind the same-input HF row-count experiment to the live TP operand trace."""
    require=lambda value,reason: None if value else fail(reason)
    require(layer in (0,1) and site in ('q','o'),'supported projection scope required')
    comparison=compare_q_repair(original,traced,hf_root,observation_only=True)
    observations={(r['lane'],r['rank']):r for r in comparison['rows']}
    controls=[read(hf_root/f'rank-{i}/RESULT.json') for i in range(8)]
    prefix=('op' if layer==0 else 'op1')+'.'+site
    rows=[]
    for lane in read(traced)['rows']:
        h=next(r['public']['boundaries'][site+'_operands'] for r in controls if r['lane']==lane['lane'])
        begin,end=h['replay']['window']
        block=h['block']
        require(begin%block==0 and (end%block==0 or end==len(lane['prompt'])),'replay window is not block aligned')
        for rank, trace in lane['prefix_traces'].items():
            first=next(iter(observations[lane['lane'],int(rank)]['different_prompt_boundaries']),None)
            t=trace[site+'_operands']
            require(t['block']==block and t['scale']==h['scale'],'projection trace block or scale mismatch')
            matching_calls=[c for c in t['calls'] if c['prompt_rows'] and c['position']==begin
                            and c['prompt_rows']==end-begin and c['scheduled']==end-begin]
            require(len(matching_calls)==1,'replay window differs from actual scheduler call')
            different={}
            for stage,blocks in h['traces'].items():
                different[stage]=[i for i,(a,b) in enumerate(zip(blocks,t['traces'][stage],strict=True)) if a!=b]
            live=t['traces']['base'][begin//block:(end+block-1)//block]
            native=h['replay']['results']['native_2d']
            grouped=h['replay']['results']['grouped_window']
            same_inputs=t['weights']==h['weights'] and not different['input']
            localized=bool(different['base']) and all(begin//block<=i<(end+block-1)//block for i in different['base'])
            rows.append(dict(lane=lane['lane'],rank=int(rank),first_observed_prompt_difference=first,
                same_operands=same_inputs,different_blocks=different,
                native_reproduces_live=same_inputs and native['blocks']==live,
                native_different_elements=native['different_elements'],native_max_abs=native['max_abs'],
                grouped_different_elements=grouped['different_elements'],
                confirmed_first_base_difference=first==prefix and same_inputs and localized and native['blocks']==live))
    return dict(schema='updated-policy-projection-row-replay-v1',layer=layer,site=site,rows=rows,training_admission=False)


def compare_q_repair(original, repaired, hf_root, *, observation_only=False, hf_shard_key_prefix=''):
    """Compare only actions actually shared by the baseline and repaired HTTP requests."""
    import math
    require=lambda value,reason: None if value else fail(reason)
    require(read(hf_root/'EXIT.json')['exit_code']==0,'incomplete HF evidence')
    hf = [read(hf_root/f'rank-{i}/RESULT.json') for i in range(8)]
    result = []
    for before, after in zip(read(original)['rows'],read(repaired)['rows'],strict=True):
        require(all(before[k]==after[k] for k in ('lane','prompt','publication_sha','adapter_sha')),
                'changed input, lane, or adapter')
        if observation_only:
            require(all(before[k]==after[k] for k in ('tokens','logprobs','raw_top1')),
                    'operator observer changed actual HTTP scores or actions')
        matching = [r for r in hf if r['lane']==before['lane']]
        require(len(matching)==4 and all(r['response_sha']==before['response'] and
                r['optimizer_updates']==0 and not r['trace_differences'] for r in matching),
                'HF baseline custody differs')
        h = matching[0]['public']
        same = before['tokens']==after['tokens']
        for rank, t in after['prefix_traces'].items():
            expected = ['decoder.input']
            for layer in range(48):
                expected.append(f'{layer}.in')
                prefix={0:'op',1:'op1'}.get(layer)
                if prefix and f'{prefix}.q.prompt' in t:
                    expected.extend(f'{prefix}.{key}' for key in ('q','k','v','attn','o'))
                expected.append(f'{layer}.post')
            def all_differences(scope):
                rows=[]
                for site in expected:
                    sharded=site.startswith(('op.','op1.')) and not site.endswith('.o')
                    left = (hf_shard_key_prefix if sharded else '')+site+('.'+rank if sharded else '')+'.'+scope
                    right = site+'.'+scope
                    require(h['boundaries'][left]['shape']==t[right]['shape'],'different diagnostic extent')
                    if h['boundaries'][left]!=t[right]:rows.append(site)
                return rows
            prompt=all_differences('prompt')
            if same:
                predictors=all_differences('predictors')
                delta=[a-b for a,b in zip(h['scores'],after['logprobs'],strict=True)]
                token,sequence=max(map(abs,delta)),abs(math.fsum(delta))
                flips=sum(a!=b for a,b in zip(h['hf_top1'],after['raw_top1'],strict=True))
                scores=dict(token_max_abs=token,sequence_abs=sequence,top1_flips=flips,
                           passes_original_numeric_gate=token<=.01 and sequence<=.01 and not flips)
            else:
                predictors=None
                scores=None
            result.append(dict(lane=after['lane'],rank=int(rank),same_actions=same,
                different_prompt_boundaries=prompt,different_predictor_boundaries=predictors,scores=scores))
    return dict(schema='updated-policy-q-repair-comparison-v1',rows=result,training_admission=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('wire','original'):
        p.add_argument('--'+name,type=Path)
    control=p.add_mutually_exclusive_group()
    control.add_argument('--hf-prefix-control',type=Path)
    control.add_argument('--hf-o-lora-b-control',type=Path)
    control.add_argument('--hf-repair-control',type=Path)
    control.add_argument('--hf-projection-replay',type=Path)
    p.add_argument('--projection-layer',type=int,choices=[0,1],default=0)
    p.add_argument('--projection-site',choices=['q','o'],default='q')
    p.add_argument('--observation-only',action='store_true')
    p.add_argument('--hf-shard-key-prefix',choices=['op.'],default='')
    a=p.parse_args()
    if a.hf_o_lora_b_control:
        result=compare_o_lora_b(a.root,a.hf_o_lora_b_control)
    elif a.hf_projection_replay:
        if a.original is None:p.error('--original required for the selected comparison')
        result=compare_projection_replay(a.original,a.root,a.hf_projection_replay,
            layer=a.projection_layer,site=a.projection_site)
    elif a.hf_prefix_control or a.hf_repair_control:
        if a.original is None:p.error('--original required for the selected comparison')
        result=(compare_prefix(a.original,a.root,a.hf_prefix_control) if a.hf_prefix_control else
                compare_q_repair(a.original,a.root,a.hf_repair_control,
                    observation_only=a.observation_only,hf_shard_key_prefix=a.hf_shard_key_prefix))
    else:
        if a.wire is None or a.original is None:p.error('--wire and --original required to prepare captured evidence')
        prepare(a.root,a.wire,a.original,a.output)
        result=None
    if result is not None:
        with a.output.open('x') as f:json.dump(result,f,indent=2)
