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


def install_rows(worker, sites):
    from retained_audio_row_precision import attach
    model = worker.model_runner.get_model()
    if hasattr(worker, '_retained_audio_rows'):
        raise ValueError('Diagnostic worker already owns audio projections')
    audit, restore = attach(model.audio_tower, sites, worker.rank)
    worker._retained_audio_rows = (audit, restore)
    return worker.rank


def close_rows(worker):
    calls, restore = worker._retained_audio_rows
    restore()
    del worker._retained_audio_rows
    return dict(rank=worker.rank, calls=dict(calls))


def decode(engine, cfg):
    from vllm import SamplingParams
    from vllm.multimodal.media.audio import AudioMediaIO
    directory = Path(cfg['input'])
    bundle = checked(directory, cfg['retained_actions'])
    result = []
    for case in bundle['cases']:
        packet = checked(directory, case['packet'])
        behavior = checked(directory, case['behavior'])['records'][case['index']]
        if packet['wire_sha256'] != behavior['wire_sha256']:
            raise ValueError('Behavior and audio request do not match')
        wire = packet['wire']
        audio, = [part['input_audio'] for msg in wire['messages'] if isinstance(msg['content'],list)
                  for part in msg['content'] if part['type']=='input_audio']
        samples, rate = AudioMediaIO().load_base64('audio/'+audio['format'],audio['data'])
        original = behavior['token_ids']
        reference = case['hf_logprobs']
        if len(reference)!=len(original) or not all(map(math.isfinite,reference)):
            raise ValueError('Missing retained HF scores')
        params = SamplingParams(seed=behavior['seed'],temperature=wire['temperature'],top_p=wire['top_p'],
                                top_k=-1 if wire['top_k']==0 else wire['top_k'],min_p=wire['min_p'],
                                repetition_penalty=wire['repetition_penalty'],presence_penalty=wire['presence_penalty'],
                                frequency_penalty=wire['frequency_penalty'],logprobs=0,max_tokens=len(original))
        req, = engine.generate([{'prompt':case['prompt'],'multi_modal_data':{'audio':(samples,rate)}}],
                               sampling_params=params,use_tqdm=False)
        if list(req.prompt_token_ids)!=behavior['prompt_token_ids']:
            raise ValueError('Retained and current prompt token IDs differ')
        out, = req.outputs
        deltas=[]
        for index,(expected,chosen,probs) in enumerate(zip(original,out.token_ids,out.logprobs)):
            if expected!=chosen:
                break
            actual=probs[chosen].logprob
            if not math.isfinite(actual):
                raise ValueError('Nonfinite selected log probability')
            deltas.append(abs(actual-reference[index]))
        result.append(dict(node=case['node'],generated=len(out.token_ids),original_tokens=len(original),
                           same_action_prefix=len(deltas),deltas=deltas,
                           all_tokens_within_limit=len(deltas)==len(original) and max(deltas)<=case['limit']))
    return result
