import json, sys, threading, subprocess, wave, struct
from http.server import BaseHTTPRequestHandler, HTTPServer
M = "Considering the limited time by the user, I have to give the solution based on the thinking directly now."
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
        if self.path == "/tokenize" and body.get("return_token_strs"):
            tail = "<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")
            strs = ["<|im_start|>system\nYou label spoken requests.<|im_end|>\n<|im_start|>user\n", "<|audio_start|>",
                    "<|audio_pad|>", "<|audio_end|>", "Name the intent.<|im_end|>\n", tail]
            code, doc = 200, {"count": len(strs), "tokens": list(range(len(strs))), "token_strs": strs}
        elif self.path == "/tokenize" and body.get("tools"):
            code, doc = 400, {"error": {"type": "BadRequestError", "code": 400, "message": "can only concatenate list (not \"str\") to list"}}
        elif self.path == "/tokenize":
            code, doc = 200, {"count": 3, "tokens": [1, 2, 3], "max_model_len": 24576}
        elif thinking is False:
            code, doc = 200, {"choices": [{"message": {"reasoning": None, "content": "alarm_set"}}]}
        elif "word word" in json.dumps(body):
            code, doc = 400, {"error": {"type": "BadRequestError", "code": 400, "message": "The decoder prompt (length 30012) is longer than the maximum model length of 24576. Make sure that `max_model_len` is no smaller than the number of text tokens plus multimodal tokens. For image inputs, the number of image tokens depends on the number of images, and possibly their aspect ratios as well."}}
        elif body.get("tools"):
            code, doc = 200, {"choices": [{"message": {"tool_calls": [{"function": {"name": "lookup_intent", "arguments": "{}"}}]}}]}
        elif body.get("max_tokens") == 1:
            code, doc = 200, {"choices": [{"message": {}}], "usage": {"prompt_tokens": 3}}
        else:
            code, doc = 200, {"choices": [{"message": {"reasoning": "Let me think " + M + "\n", "content": "ok"}}]}
        raw = json.dumps(doc).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
srv = HTTPServer(("127.0.0.1", 18299), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with wave.open("/tmp/a.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(struct.pack("<100h", *[0]*100))
sys.exit(subprocess.call([sys.executable, "/mnt/cephfs/xingchao/omni/work/scripts/npu_smoke_thinking.py", "--base-url", "http://127.0.0.1:18299", "--audio", "/tmp/a.wav", "--out", "/tmp/smoke-out"]))
