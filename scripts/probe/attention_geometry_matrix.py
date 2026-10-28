"""Effective attention geometry of the CPU stack, by batch shape and by forced flag (synthetic text only).

Cases: batch-one (sdpa may skip the mask and consult the module's is_causal flag) and a padded batch of two
(the mask is materialised). Each case is measured with the model as loaded (remote code sets is_causal=False)
and again with every text layer forced to is_causal=True.

usage (port_py.sh, system python): attention_geometry_matrix.py <model-dir>
"""
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "32"
sys.modules["torch_npu"] = None
import torch

torch.set_num_threads(32)
from speech_aware_evidence_acquisition.core.speech_embedding import load_stack

processor, model, _ = load_stack(Path(sys.argv[1]), local_files_only=True, runtime="cpu-float32")
tok = processor.tokenizer
text_model = model.model
short = "passage: switch the kitchen lights"
long = "passage: switch the kitchen lights to blue please right now"
filler = "passage: a much longer unrelated sentence about the weather forecast for the coming week in the mountains"


def hidden(texts, index):
    batch = tok(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    length = int(batch["attention_mask"][index].sum())
    return out[index, :length]


def prefix_diff(batch_of):
    n = tok(short, return_tensors="pt")["input_ids"].shape[1]
    a = hidden(batch_of(short), 0)[:n]
    b = hidden(batch_of(long), 0)[:n]
    return (a - b).abs().max().item()


for forced in (None, True):
    if forced is not None:
        for layer in text_model.layers:
            layer.self_attn.is_causal = forced
    flags = sorted({layer.self_attn.is_causal for layer in text_model.layers})
    print("padding side", tok.padding_side, "forced", forced, "flags", flags, flush=True)
    for name, batch_of in (("batch-one", lambda t: [t]), ("padded-batch-2", lambda t: [t, filler])):
        diff = prefix_diff(batch_of)
        print(f"  {name:15s} max|prefix hidden diff| {diff:.6f} -> {'causal' if diff < 1e-3 else 'bidirectional'}", flush=True)
