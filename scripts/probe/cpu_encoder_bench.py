"""CPU throughput and dtype drift of the pinned Nemotron stack on synthetic inputs (no dataset content).

usage (port_py.sh): cpu_encoder_bench.py <model-dir> <threads>
"""

import sys
import time

sys.modules["torch_npu"] = None  # CPU only: no Ascend driver in this container

import faulthandler

faulthandler.dump_traceback_later(90, repeat=True)

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from speech_aware_evidence_acquisition.core.speech_embedding import encode, render

model_dir, threads = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(threads)
print("threads", torch.get_num_threads(), torch.__config__.parallel_info().splitlines()[:6], flush=True)
template = open(f"{model_dir}/additional_chat_templates/sentence_transformers.jinja", encoding="utf-8").read()
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
passages = ["passage: set an alarm for the morning meeting tomorrow",
            "passage: Intent example. Activation boundary: a request to change the colour or brightness of a lamp."
            " Confusable exclusions: switching it off entirely is a different operation." * 3]
rng = np.random.default_rng(0)
waves = [(0.05 * rng.standard_normal(16000 * s)).astype(np.float32) for s in (3, 8)]
vectors = {}
for dtype in (torch.float32, torch.bfloat16):
    t0 = time.time()
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, dtype=dtype,
                                      attn_implementation="sdpa").eval()
    print(f"{dtype} load {time.time() - t0:.1f}s", flush=True)
    out = []
    for text in passages:
        rendered = render(processor, template, [{"type": "text", "text": text}])
        encode(processor, model, [rendered], None, device="cpu")  # warm-up
        t0 = time.time()
        out.append(encode(processor, model, [rendered], None, device="cpu")[0])
        print(f"{dtype} text {len(processor.tokenizer(rendered)['input_ids'])} tok {time.time() - t0:.2f}s", flush=True)
    for wave in waves:
        rendered = render(processor, template, [{"type": "text", "text": "query: "}, {"type": "audio"}])
        t0 = time.time()
        out.append(encode(processor, model, [rendered], [wave], device="cpu", max_audio_samples=len(wave),
                          require_complete_audio=True)[0])
        print(f"{dtype} audio {len(wave) / 16000:.0f}s {time.time() - t0:.2f}s", flush=True)
    vectors[dtype] = np.array(out)
    del model
cos = (vectors[torch.float32] * vectors[torch.bfloat16]).sum(1)
print("fp32 vs bf16 cosine per input:", np.round(cos, 5).tolist())
