"""Compare original actions to natural decode from the same frozen audio policy."""
import hashlib
import json
import math
from pathlib import Path


def checked(directory, ref):
    filename = ref['file']
    if Path(filename).name != filename:
        raise ValueError('Expected a local replay input')
    data = (directory / filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('Retained input changed: ' + filename)
    return json.loads(data)


def install_rows(worker, sites, control=None, gate_candidate_layers=None,
                 attention_candidate_layers=None, down_candidate_layers=None,
                 audio_attention_layers=None, router_layers=None):
    from retained_audio_row_precision import attach
    model = worker.model_runner.get_model()
    if hasattr(worker, '_retained_audio_rows'):
        raise ValueError('Diagnostic worker already owns audio projections')
    audit, restore = attach(model.audio_tower, sites, worker.rank)
    if control is not None:
        # Only in a fresh diagnostic engine. The real encoder still executes;
        # substitute its output before split/merge and normal cache insertion.
        import torch
        from safetensors.torch import load_file
        path = Path(control['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != control['tensor_file_sha256']:
            raise ValueError('HF audio reference changed')
        ref = load_file(str(path), device='cpu')['audio_out']
        if ref.dtype != torch.bfloat16 or ref.ndim != 2 or not torch.isfinite(ref).all():
            raise ValueError('invalid sealed HF audio output')
        receipt = dict(reference=control['tensor_file_sha256'], observations=[])
        def replace(module, args, output):
            if output.dtype != ref.dtype or output.shape != ref.shape:
                raise ValueError('audio intervention shape/dtype differs')
            natural = output.detach().cpu().contiguous()
            receipt['observations'].append(dict(
                native_sha256=hashlib.sha256(natural.view(torch.uint8).numpy().tobytes()).hexdigest(),
                unequal=int(torch.count_nonzero(natural != ref)),
                max_abs=float((natural.float()-ref.float()).abs().max())))
            return ref.to(output.device)
        worker._retained_audio_control = (receipt, model.audio_tower.register_forward_hook(replace))
    worker._retained_audio_rows = (audit, restore)
    if audio_attention_layers is not None:
        if control is not None:
            raise ValueError('audio attention candidate requires the original tower output')
        from retained_audio_attention_precision import attach
        worker._retained_audio_attention = attach(model.audio_tower, audio_attention_layers, worker.rank)
    if gate_candidate_layers is not None:
        from retained_expert_candidate import attach as attach_candidate
        worker._retained_gate_candidate = attach_candidate(model, gate_candidate_layers)
    if attention_candidate_layers is not None:
        from retained_attention_candidate import attach_vllm
        worker._retained_attention_candidate = attach_vllm(model, attention_candidate_layers)
    if down_candidate_layers is not None:
        from retained_expert_candidate import attach
        worker._retained_down_candidate = attach(model, down_candidate_layers, stage='down')
    if router_layers:
        from retained_router_candidate import attach as attach_router
        worker._retained_router = attach_router(model.language_model.model,router_layers,side="vl")
    return worker.rank


def close_rows(worker):
    calls, restore = worker._retained_audio_rows
    restore()
    del worker._retained_audio_rows
    result = dict(rank=worker.rank, calls=dict(calls))
    if hasattr(worker, '_retained_audio_attention'):
        result['audio_attention'] = worker._retained_audio_attention()
        del worker._retained_audio_attention
    if hasattr(worker, '_retained_router'):
        result['router']=worker._retained_router()
        del worker._retained_router
    if hasattr(worker, '_retained_down_candidate'):
        result['down_candidate'] = worker._retained_down_candidate()
        del worker._retained_down_candidate
    if hasattr(worker, '_retained_gate_candidate'):
        result['gate_candidate'] = worker._retained_gate_candidate()
        del worker._retained_gate_candidate
    if hasattr(worker, '_retained_attention_candidate'):
        result['attention_candidate'] = worker._retained_attention_candidate()
        del worker._retained_attention_candidate
    if hasattr(worker, '_retained_audio_control'):
        receipt, hook = worker._retained_audio_control
        hook.remove()
        del worker._retained_audio_control
        if not receipt['observations']:
            raise ValueError('audio reference never used in decode')
        result['intervention'] = receipt
    return result


def score_retained_prefill(engine, case, record, audio, sampling_params):
    """Score every original action in one prefill, regardless of sampled tokens."""
    base, actions = record['prompt_token_ids'], record['token_ids']
    # Pass token IDs *before* multimodal expansion; the model processor owns
    # expansion. Appending decoded text could merge tokens across the join.
    compact = list(engine.get_tokenizer().encode(case['prompt'], add_special_tokens=False))
    request = {'prompt_token_ids': compact + actions, 'multi_modal_data': {'audio': audio}}
    params = sampling_params(temperature=1., top_p=1., top_k=-1, min_p=0.,
                             prompt_logprobs=1, max_tokens=1, seed=record['seed'])
    out, = engine.generate([request], sampling_params=params, use_tqdm=False)
    if list(out.prompt_token_ids) != base + actions:
        raise ValueError('Teacher-forced expanded IDs differ from original context/actions')
    if len(out.outputs) != 1 or len(out.outputs[0].token_ids) != 1:
        raise ValueError('Unexpected prefill probe output count')
    probs = out.prompt_logprobs
    if probs is None or len(probs) != len(out.prompt_token_ids):
        raise ValueError('Incomplete prompt scores')
    per_action = [probs[len(base)+i] for i in range(len(actions))]
    scores = [p[token].logprob for p,token in zip(per_action,actions,strict=True)]
    argmax = [next((token for token,v in p.items() if v.rank==1),None) for p in per_action]
    if not scores or not all(map(math.isfinite,scores)):
        raise ValueError('Nonfinite or missing teacher-forced scores')
    deltas = [a-b for a,b in zip(scores,case['hf_logprobs'],strict=True)]
    return dict(mode='teacher_forced_prefill', actions_scored=len(scores), extra_generated=1,
                scores=scores, max_abs=max(map(abs,deltas)), sequence_abs=abs(math.fsum(deltas)),
                signed_deltas=deltas, observed_raw_top1=argmax,
                prefix_cache_read_skipped_for_prompt_scores=params.skip_reading_prefix_cache)


def decode(engine, cfg):
    from vllm import SamplingParams
    from vllm.multimodal.media.audio import AudioMediaIO
    directory = Path(cfg['input'])
    bundle = checked(directory, cfg['retained_actions'])
    result = []
    pending_prefill = []
    def trace_begin(case, record, mode):
        trace=cfg.get('trace_predictor')
        if not trace or trace['node']!=case['node']:return False
        ordinal=trace['action_ordinal']
        if not 1<=ordinal<=len(record['token_ids']):raise ValueError('invalid target action')
        position=len(record['prompt_token_ids'])+ordinal-2+trace['rope_offset']
        ranks=engine.collective_rpc('omni_predictor_begin',
            args=(mode,position,trace.get('attention_layer'),trace.get('prefix_origin'),
                  trace.get('moe_origin',False),trace.get('expert_operands',False),
                  trace.get('prefix_scan'),trace.get('origin_layers',2),
                  trace.get('attention_shape_shadow',False),trace.get('moe_sites')),timeout=240)
        if sorted(ranks)!=list(range(4)):raise ValueError('missing trace worker')
        return True
    for case in bundle['cases']:
        packet = checked(directory, case['packet'])
        behavior = checked(directory, case['behavior'])['records'][case['index']]
        if packet['wire_sha256'] != behavior['wire_sha256']:
            raise ValueError('Behavior and audio request do not match')
        wire = packet['wire']
        audio, = [part['input_audio'] for msg in wire['messages'] if isinstance(msg['content'],list)
                  for part in msg['content'] if part['type']=='input_audio']
        if 'audio_control' in cfg:
            import base64
            if hashlib.sha256(base64.b64decode(audio['data'], validate=True)).hexdigest() != cfg['audio_control']['audio_sha256']:
                raise ValueError('audio source differs from HF control')
        samples, rate = AudioMediaIO().load_base64('audio/'+audio['format'],audio['data'])
        original = behavior['token_ids']
        reference = case['hf_logprobs']
        if len(reference)!=len(original) or not all(map(math.isfinite,reference)):
            raise ValueError('Missing retained HF scores')
        params = SamplingParams(seed=behavior['seed'],temperature=wire['temperature'],top_p=wire['top_p'],
                                top_k=-1 if wire['top_k']==0 else wire['top_k'],min_p=wire['min_p'],
                                repetition_penalty=wire['repetition_penalty'],presence_penalty=wire['presence_penalty'],
                                frequency_penalty=wire['frequency_penalty'],logprobs=1,max_tokens=len(original))
        tracing=trace_begin(case,behavior,'decode')
        request = {'prompt':case['prompt'],'multi_modal_data':{'audio':(samples,rate)}}
        cache_pair = cfg.get('prefix_cache_pair') == case['node']
        if cache_pair:
            request['cache_salt'] = 'retained-native-parity-' + packet['wire_sha256']
        req, = engine.generate([request],
                               sampling_params=params,use_tqdm=False)
        if list(req.prompt_token_ids)!=behavior['prompt_token_ids']:
            raise ValueError('Retained and current prompt token IDs differ')
        out, = req.outputs
        deltas=[]
        scores=[]
        raw_top1=[]
        for index,(expected,chosen,probs) in enumerate(zip(original,out.token_ids,out.logprobs)):
            if expected!=chosen:
                break
            actual=probs[chosen].logprob
            if not math.isfinite(actual):
                raise ValueError('Nonfinite selected log probability')
            scores.append(actual)
            raw_top1.append(next((token for token,v in probs.items() if v.rank==1),None))
            deltas.append(abs(actual-reference[index]))
        result.append(dict(node=case['node'],generated=len(out.token_ids),original_tokens=len(original),
                           same_action_prefix=len(deltas),deltas=deltas, matching_prefix_scores=scores,
                           matching_prefix_raw_top1=raw_top1,
                           all_tokens_within_limit=len(deltas)==len(original) and max(deltas)<=case['limit']))
        if cache_pair:
            warm, = engine.generate([request], sampling_params=params,use_tqdm=False)
            if list(warm.prompt_token_ids) != behavior['prompt_token_ids']:
                raise ValueError('warm cache changed the expanded prompt IDs')
            chosen, = warm.outputs
            matching, top1 = [], []
            for expected, actual, lp in zip(original,chosen.token_ids,chosen.logprobs):
                if expected != actual:
                    break
                value = lp[actual].logprob
                if not math.isfinite(value):
                    raise ValueError('nonfinite warm action score')
                matching.append(value)
                top1.append(next((i for i,v in lp.items() if v.rank==1),None))
            result[-1]['cache_pair'] = dict(cold_cached_tokens=req.num_cached_tokens,
                warm_cached_tokens=warm.num_cached_tokens, prompt_tokens=len(req.prompt_token_ids),
                generated=len(chosen.token_ids), scores=matching, raw_top1=top1)
        if tracing:
            if len(deltas)<cfg['trace_predictor']['action_ordinal']:
                raise ValueError('traced action differs from original')
            result[-1]['natural_trace']=engine.collective_rpc('omni_predictor_finish',timeout=240)
        if cfg.get('score_retained_prefill'):
            pending_prefill.append((case, behavior, (samples,rate)))
    # Run these after all natural decodes so the additional requests cannot
    # warm an encoder/prefix cache for the paired natural arm.
    for row, (case, record, audio) in zip(result, pending_prefill):
        tracing=trace_begin(case,record,'prefill')
        row['forced_prefill'] = score_retained_prefill(engine, case, record, audio, SamplingParams)
        if tracing:
            row['paired_trace']=engine.collective_rpc('omni_predictor_finish',timeout=240)
    return result
