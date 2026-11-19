"""Trace installed HF/vLLM audio stem forward methods on retained BF16 input."""
import base64
import hashlib
import inspect
import json
from pathlib import Path


def analyze(c, device):
    import torch
    from torch import nn
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder as HF
    from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder as VL
    from audio_conv_precision import attach
    root = Path(c['output']).parent
    model = Path(c['model'])
    data = json.loads((root / 'features.json').read_bytes())
    buf = base64.b64decode(data['bf16'], validate=True)
    assert hashlib.sha256(buf).hexdigest() == data['sha256']
    feat = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(data['shape']).to(f'npu:{device}')
    lengths = torch.tensor([feat.shape[-1]], device=feat.device)
    cfg = json.loads((model/'config.json').read_text())['thinker_config']['audio_config']
    index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    names = ('conv2d1', 'conv2d2', 'conv2d3')
    weights = {}
    for name in names:
        params = {}
        for param in ('weight', 'bias'):
            key = 'thinker.audio_tower.' + name + '.' + param
            with safe_open(str(model/index[key]), framework='pt', device='cpu') as f:
                params[param] = f.get_tensor(key)
        weights[name] = params
    class StopAtProjection(Exception):
        pass
    def trace(cls, native_format, installed_repair):
        torch.npu.config.allow_internal_format = native_format
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        for field in ('n_window', 'n_window_infer', 'conv_chunksize'):
            setattr(m, field, cfg[field])
        seen = {}
        for name in names:
            w = weights[name]['weight']
            conv = nn.Conv2d(w.shape[1], w.shape[0], 3, 2, padding=1, dtype=w.dtype)
            conv.load_state_dict(weights[name])
            setattr(m, name, conv.to(feat.device))
            def obs(mod, args, result, key=name):
                seen[key+'.in'] = args[0].detach().cpu()
                seen[key+'.out'] = result.detach().cpu()
            conv.register_forward_hook(obs)
        m.conv_out = nn.Identity()
        def stop(_, args):
            seen['projection.in'] = args[0].detach().cpu()
            raise StopAtProjection
        m.conv_out.register_forward_pre_hook(stop)
        restore = None
        if installed_repair:
            audit, restore = attach(m)
        try:
            with torch.no_grad(), torch.inference_mode(cls is VL), torch.autocast('npu', dtype=torch.bfloat16, enabled=cls is HF):
                fn = inspect.unwrap(cls.forward)
                if cls is VL:
                    fn(m, feat, lengths, m._get_cnn_output_lengths(lengths))
                else:
                    fn(m, feat, lengths)
        except StopAtProjection:
            pass
        finally:
            if restore:
                restore()
        assert len(seen) == 7
        return seen
    def metrics(left, right):
        result = {}
        for key, a in left.items():
            b = right[key]
            assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
            result[key] = dict(shape=list(a.shape), unequal=int((a != b).sum()), max_abs=float((a.float()-b.float()).abs().max()))
        return result
    reference = trace(HF, False, False)
    existing = metrics(reference, trace(VL, True, True))
    control = metrics(reference, trace(VL, False, False))
    peak = torch.npu.max_memory_allocated()
    assert peak < c['max_device_bytes']
    return dict(schema='retained-audio-stem-operator-replay-v1', existing=existing,
                internal_format_disabled_control=control, feature_digest=data['sha256'],
                source_digests={cls.__module__:hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() for cls in (HF,VL)},
                first_differing_boundary=next((k for k,v in existing.items() if v['unequal']),None),
                peak_allocated_bytes=peak, complete_original_forward=False)
