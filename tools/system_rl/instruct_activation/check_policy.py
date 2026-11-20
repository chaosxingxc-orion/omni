"""Use a real audio wire to verify local TP4 policy activation on each lane."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from io_utils import doc, pin, require, save, storage
from measure_existing_services import contact, idle, metrics


def receipt(lane,name):
    path=Path(lane['policy_state'])/name
    if 'container' not in lane:
        ref=pin(path)
        return doc(ref),ref
    container=lane['container']
    require(re.fullmatch('[a-z0-9][a-z0-9-]+',container),'invalid local container name')
    prefix='successor/' if lane.get('successor') else ''
    raw=subprocess.check_output(['docker','exec',container,'cat','/runtime/state/policy/'+prefix+name],timeout=15)
    return json.loads(raw),dict(container=container,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))



def stopped_terminal(lane, root):
    """An exact stopped owner plus a known terminal HTTP response releases its lane."""
    container=lane.get('container')
    if not container or (root/lane['name']/'UNKNOWN.json').exists():
        return None
    path=root/lane['name']/'RESPONSE.json'
    if not path.is_file():
        return None
    response=doc(pin(path))
    status=response.get('status')
    if (type(status) is not int or not 200<=status<=599 or
            hashlib.sha256(response['raw'].encode()).hexdigest()!=response['response_sha256']):
        return None
    info=json.loads(subprocess.check_output(['docker','inspect',container],timeout=10))
    if len(info)!=1 or info[0]['Name']!='/'+container:
        return None
    row=info[0]
    if row['State']['Status']!='exited' or row['State']['Running']:
        return None
    return dict(container_id=row['Id'],state=row['State']['Status'],
                finished_at=row['State']['FinishedAt'],response=pin(path),http_status=status)

def check(config):
    root=Path(config['output'])
    storage(root,1024**3)
    wire=doc(config['wire'])
    require(wire['chat_template_kwargs']['enable_thinking'] is False and wire['return_token_ids']
            and wire['logprobs'] and wire['top_logprobs']==0 and not wire['stream'],'real sampled audio wire required')
    require(not root.exists(),'prior diagnostic attempt: reconcile instead of replaying')
    lanes=config['lanes']
    require(lanes and len({lane['endpoint'] for lane in lanes})==len(lanes) and len({lane['name'] for lane in lanes})==len(lanes),'independent named lanes required')
    with Path(config['guard']).open('rb') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for lane in lanes:
            require(lane['endpoint'].startswith('http://127.0.0.1:'),'loopback only')
            idle(metrics(lane['endpoint']))
        if config.get('switch'):
            drained=pin(config['switch']['drained_path'])
            terminal=doc(drained)
            require(terminal['guards_releasable'] and not list(Path(drained['path']).parent.rglob('UNKNOWN.json')),
                    'predecessor requests not closed')
            for lane in lanes:
                require(lane['successor'],'successor receipts required')
                marker=json.dumps(config['switch']['policies']).encode()
                subprocess.run(['docker','exec','-i',lane['container'],'python3','-c',
                    "import sys,pathlib,os;p=pathlib.Path('/runtime/state/policy/SWITCH-REQUEST.json');f=p.with_suffix('.pending').open('xb');f.write(sys.stdin.buffer.read());f.flush();os.fsync(f.fileno());f.close();os.link(f.name,p)"],input=marker,check=True)
        save(root/'INTENT.json',dict(lanes=lanes,wire=config['wire'],automatic_retry=False,training_use=False))
        started=time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
                jobs=[pool.submit(contact,root/lane['name'],lane['endpoint'],wire,config['request_timeout_seconds'])
                      for lane in lanes]
                replies=[job.result() for job in jobs]
            result=[]
            for lane,reply in zip(lanes,replies,strict=True):
                activation,act_ref=receipt(lane,'ACTIVATION.json')
                cache,cache_ref=receipt(lane,'CACHE-POLICY.json')
                precision,precision_ref=receipt(lane,'PRECISION.json')
                if 'container' in lane:
                    require(activation['service_identity']['container_name']==lane['container'],'wrong service container')
                require(activation['endpoint']==lane['endpoint']+'/v1/chat/completions' and
                        activation['weights_ref']['sha256']==config['adapter_sha256'],'wrong actual adapter version')
                require({r['rank'] for r in activation['workers']}==set(range(lane.get('world_size',len(activation['service_identity']['physical_devices'])))) and
                        all(r['compared_original_keys']==config.get('adapter_tensors',len(r['loaded_tensors']))==len(r['loaded_tensors'])>0 for r in activation['workers']),'incomplete actual worker readback')
                require(cache['reset_succeeded'] and cache['kv_cache'] and cache['prefix_cache'],'cache receipt required')
                require({r['rank'] for r in precision['workers']}==set(range(lane.get('world_size',len(activation['service_identity']['physical_devices'])))),'precision workers absent')
                result.append(dict(activation_ref=act_ref,cache_ref=cache_ref,precision_ref=precision_ref,**reply))
            save(root/'RESULT.json',dict(lanes=result,elapsed_seconds=time.monotonic()-started,training_use=False))
        finally:
            # A timeout may leave engine work unresolved. Keep the original
            # guard until the service itself shows no running or waiting work.
            while True:
                try:
                    stopped={}
                    for lane in lanes:
                        try:
                            idle(metrics(lane['endpoint']))
                        except (OSError,ValueError):
                            terminal=stopped_terminal(lane,root)
                            if terminal is None:
                                raise
                            stopped[lane['name']]=terminal
                    break
                except (OSError,ValueError,subprocess.SubprocessError):time.sleep(10)
            save(root/'DRAINED.json',dict(guards_releasable=True,automatic_retry=False,
                                        stopped_owners=stopped))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    check(doc(pin(p.parse_args().config)))
