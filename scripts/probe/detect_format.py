"""vLLM 0.30's own chat-template content-format detection for the Thinking template (offline, no engine)."""
import json
import sys
from pathlib import Path

from vllm.renderers.hf import _detect_content_format

source = json.loads(Path(sys.argv[1], "chat_template.json").read_text())["chat_template"]
print("detected:", _detect_content_format(source, default="string"))
