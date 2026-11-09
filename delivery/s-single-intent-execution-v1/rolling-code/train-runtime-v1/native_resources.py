"""Hash-bound immutable tiktoken resource; initialize before generation lease."""
from pathlib import Path
import hashlib
import os
import time

URL='https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'
EXPECTED='223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'
CACHE_KEY=hashlib.sha1(URL.encode()).hexdigest()

def prepare_native_resources():
    start=time.monotonic()
    cache=Path(__file__).resolve().parents[1]/'resources/tiktoken-cache'
    raw=(cache/CACHE_KEY).read_bytes()
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=EXPECTED:raise ValueError('immutable cl100k cache SHA mismatch')
    os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
    import tiktoken
    enc=tiktoken.get_encoding('cl100k_base')
    if enc.name!='cl100k_base' or enc.n_vocab!=100277:raise ValueError('encoding identity mismatch')
    return dict(status='LOCAL_NATIVE_ENCODING_READY',cache_sha256=actual,cache_key=CACHE_KEY,
                encoding=enc.name,vocab=enc.n_vocab,seconds=time.monotonic()-start,network_calls=0)
