"""Profile an isolated real request after a streamed output token, under service guards."""
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from io_utils import canonical, doc, pin, require, save, storage
from measure_existing_services import idle, metrics, service_identity


def lane(s, cfg, wire):
    root = Path(cfg['output']) / s['name']
    body = dict(wire, stream=True, stream_options={'include_usage': True},
                cache_salt=cfg['measurement_id'])
    if 'stock_adapter_alias' in s:
        body['model'] = s['stock_adapter_alias']
    data = canonical(body)
    save(root/'INTENT.json', dict(sha256=hashlib.sha256(data).hexdigest(),
         source=cfg['wire'], automatic_retry=False, started=time.time()))
    def post(path, data=None):
        return urllib.request.urlopen(urllib.request.Request(s['endpoint']+path,
            data=data, headers={'Content-Type':'application/json'}, method='POST'),
            timeout=cfg['timeout'])
    done = started = False
    content = []
    usage = reason = None
    first = None
    try:
        with post('/v1/chat/completions', data) as resp:
            for line in resp:
                if not line.startswith(b'data:'): continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    done = True
                    break
                event = json.loads(payload)
                require('error' not in event, 'stream failed: '+str(event.get('error')))
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    reason = choice.get('finish_reason') or reason
                    token = choice.get('delta', {}).get('content')
                    if token:
                        content.append(token)
                        if first is None:
                            first = time.time()
                            with post('/start_profile') as p:
                                require(p.status == 200,'start_profile failed')
                                started = True
        require(done and reason == 'stop' and usage and usage['completion_tokens']>1,
                'incomplete decode')
        save(root/'RESPONSE.json',dict(first_content_epoch=first,end_epoch=time.time(),
             chunks=len(content), response=''.join(content),usage=usage,finish_reason=reason))
    except Exception as e:
        save(root/'FAILED.json',dict(error=repr(e),first_content_epoch=first,retry=False))
        raise
    finally:
        if started:
            with post('/stop_profile') as p: require(p.status==200,'stop_profile failed')


def run(cfg):
    root=Path(cfg['output']); template=doc(cfg['service_config']); wire=doc(cfg['wire'])
    require(template['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'boot changed')
    require(wire['model']=='qwen3-omni-30b-a3b-instruct' and not wire.get('stream') and
            all(s['endpoint'].startswith('http://127.0.0.1:') for s in template['services']), 'business input or endpoint')
    storage(root,2*1024**3)
    with contextlib.ExitStack() as locks:
        for guard in sorted(template['guards']):
            fcntl.flock(locks.enter_context(open(guard,'rb')),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for s in template['services']:
            service_identity(s);idle(metrics(s['endpoint']))
        require(not (root/'STARTED.json').exists(),'existing request - reconcile first')
        save(root/'STARTED.json',dict(timestamp=time.time(),wire=cfg['wire'],training_use=False))
        try:
            with concurrent.futures.ThreadPoolExecutor(len(template['services'])) as pool:
                results=[pool.submit(lane,s,cfg,wire) for s in template['services']]
                for f in results:f.result()
            save(root/'RESULT.json',dict(stream_complete=True,verify_actual_shapes_offline=True))
        finally:
            while True:
                try:
                    for s in template['services']: idle(metrics(s['endpoint']))
                    break
                except Exception:time.sleep(10)
            save(root/'DRAINED.json',dict(time=time.time()))

if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_bytes()))
