"""Processor-only probe (no model weights, synthetic text): which side does the processor pad on?"""
import sys
sys.modules["torch_npu"] = None  # CPU-only probe: never initialise an accelerator backend
from transformers import AutoProcessor

for path in sys.argv[1:]:
    proc = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code="nemotron" in path)
    batch = proc(text=["alpha bravo", "alpha bravo charlie delta echo foxtrot"], return_tensors="pt", padding=True)
    print(path.rsplit("/", 1)[-1], "tokenizer.padding_side=", proc.tokenizer.padding_side,
          "short-row mask=", batch["attention_mask"][0].tolist())
