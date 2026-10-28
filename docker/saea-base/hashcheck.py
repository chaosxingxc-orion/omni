import sys
sys.path.insert(0, "/opt/umbrella/studies/speech-aware-evidence-acquisition/src")
from pathlib import Path
from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
R = Path("/opt/umbrella/studies/speech-aware-evidence-acquisition")
SPLIT = "89b178d6b1ef46d9ca820ae54349e3d2099c60cde6d6b753ef9f66137c310028"
for model in ("qwen3-omni-llamacpp-q4km-e001", "qwen3-omni-llamacpp-q4km"):
    c = compose(R, model, "earnings22-dev-subset10", "bare-core-ojw", "r0-smoke")
    print(model)
    print("  config_hash  ", c.config_hash)
    print("  protocol_hash", protocol_hash(c.config_hash, str(c.values["arm"]), SPLIT))
print("memo: config 56f5a0d42a7b558c  protocol aae46aee1b3e24db")
