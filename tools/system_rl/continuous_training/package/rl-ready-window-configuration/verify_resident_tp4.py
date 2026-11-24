"""Read back the locked TP4 provider and its closed audio protocol under its guard."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
from urllib.request import urlopen


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(ref):
    raw = Path(ref['path']).read_bytes()
    require(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'],
            'file changed: '+ref['path'])
    return json.loads(raw)


def observed(config):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    service = checked(config['service_config'])
    endpoint = f"http://127.0.0.1:{service['port']}"
    require(service['host_boot_id'] == boot and service['served_model_name'] == config['model']
            and service['client_chat_template_kwargs'] == {'enable_thinking': False}
            and service['serve_args'].count('--enable-prefix-caching') == 1,
            'locked provider service configuration differs')
    with Path(service['guard']).open('rb') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info, = json.loads(subprocess.check_output(['docker','inspect',config['container']],timeout=15))
        physical = sorted(int(Path(row['PathOnHost']).name.removeprefix('davinci'))
                          for row in info['HostConfig']['Devices']
                          if Path(row['PathOnHost']).name.removeprefix('davinci').isdigit())
        mounts = {row['Destination']:row for row in info['Mounts']}
        bound = mounts['/activation']
        require(info['State']['Running'] and info['Image'] == service['image']
                and physical == service['cards'] and info['Config']['Cmd'][-2:] == ['/activation/service.py','serve']
                and not bound['RW'] and Path(bound['Source'])/'config.json' == Path(config['service_config']['path']),
                'live service container does not match locked configuration')
        for label in ('bundle','update_mount','checkpoint_mount'):
            require(mounts[service['policy_overlay'][label]]['RW'] is False,'unsealed provider mount')
        with urlopen(endpoint+'/health',timeout=10) as reply:
            require(reply.status == 200,'provider health differs')
        with urlopen(endpoint+'/v1/models',timeout=10) as reply:
            models = json.load(reply)['data']
        with urlopen(endpoint+'/metrics',timeout=10) as reply:
            metrics = reply.read().decode()
        for key in ('running','waiting'):
            rows = [float(row.rsplit(' ',1)[1]) for row in metrics.splitlines()
                    if row.startswith('vllm:num_requests_'+key+'{')]
            require(rows and sum(rows)==0,'service has active requests')
        receipts = {}
        for name, expected in config['receipts'].items():
            path = Path(config['state_directory'])/name
            raw = subprocess.check_output(['docker','exec',info['Id'],'cat',str(path)],timeout=15)
            require(hashlib.sha256(raw).hexdigest()==expected,'live policy evidence changed: '+name)
            receipts[name] = raw.decode()
        act = json.loads(receipts['ACTIVATION.json'])
        prec = json.loads(receipts['PRECISION.json'])
        cache = json.loads(receipts['CACHE-POLICY.json'])
        require(act['source_boot']==boot and act['service_identity']['physical_devices']==physical
                and act['service_identity']['image']==info['Image'] and act['fixture'] is False
                and act['base'] is False and act['endpoint']==endpoint+'/v1/chat/completions'
                and act['weights_ref']['sha256']==config['adapter_sha256']
                and act['policy_version']==config.get('policy_version','checkpoint-sha256:'+config['adapter_sha256'])
                and act['policy_version'] in {kind+'-sha256:'+config['adapter_sha256'] for kind in ('initial','checkpoint')}
                and {row['id'] for row in models}=={config['model'],act['alias']},
                'live adapter, API or container identity changed')
        require(len(act['workers'])==4 and {row['rank'] for row in act['workers']}==set(range(4))
                and all(row['compared_original_keys']==384 and row['sha256']==config['adapter_sha256']
                        and row['device_buffers'] and all(buffer['exact'] is True
                            for buffer in row['device_buffers'].values()) for row in act['workers']),
                'complete NPU adapter readback required')
        require(prec['actual_device_execution'] is True and {row['rank'] for row in prec['workers']}==set(range(4))
                and cache['kv_cache'] is True and cache['prefix_cache'] is True
                and cache['reset_succeeded'] is True and cache['policy_ref']==act['policy_ref']
                and cache['precision_ref']==prec['specification_ref'],'precision or cache evidence differs')
        round_result = checked(config['round_result'])
        require(checked(config['drained'])['guards_releasable'] is True,'measurement lacks drain receipt')
        row = next(row for row in round_result['requests']
                   if row['receipt_ref']['path']==config['response_ref']['path'])
        require(row['receipt_ref']==config['response_ref'],'response is outside the closed measurement')
        response = checked(config['response_ref'])
        raw = response['raw'].encode()
        body = json.loads(raw)
        choice, = body['choices']
        tokens = choice['token_ids']
        probs = choice['logprobs']['content']
        require(response['status']==200 and hashlib.sha256(raw).hexdigest()==response['response_sha256']
                and body['model']==config['model'] and body['prompt_token_ids']
                and tokens and len(tokens)==len(probs)
                and all(p['token']==f'token_id:{t}' and math.isfinite(p['logprob']) and 0>=p['logprob']>-9999
                        for t,p in zip(tokens,probs,strict=True)), 'closed wire lacks exact sampled token probabilities')
        return dict(schema='resident-tp4-startup-observation-v1', ready=True, container_id=info['Id'],
                    source_boot=boot, devices=physical, guard_inode=guard.fileno() and __import__('os').fstat(guard.fileno()).st_ino,
                    adapter_sha256=config['adapter_sha256'], receipt_bytes=receipts, protocol_prompt_tokens=len(body['prompt_token_ids']),
                    protocol_action_tokens=len(tokens), response_ref=config['response_ref'],
                    round_ref=config['round_result'], service_config_ref=config['service_config'], endpoint=act['endpoint'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--on-host', action='store_true')
    opts = parser.parse_args()
    if opts.on_host:
        print(json.dumps(observed(json.load(__import__('sys').stdin))))
        return
    config = json.loads(Path(opts.config).read_bytes())
    source = Path(__file__).read_text()
    argv = ['python3','-B','-c',source,'--on-host']
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',config['host'],shlex.join(argv)],
                            input=json.dumps(config['observation']),text=True,capture_output=True,
                            timeout=config['timeout_seconds'],check=True)
    report = json.loads(result.stdout)
    from dependencies import ROOT, load
    from run_package import pin, write
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    commit = load(ROOT/'rollout-capture-contract/capture.py','resident_tp4_raw_receipt').commit
    expected = config['observation']['receipts']
    receipts = report.pop('receipt_bytes')
    require(set(receipts)==set(expected)=={'ACTIVATION.json','PRECISION.json','CACHE-POLICY.json'},
            'unexpected provider receipt set')
    state = {}
    for name, raw in receipts.items():
        payload = raw.encode()
        require(hashlib.sha256(payload).hexdigest() == expected[name], 'returned provider bytes changed: '+name)
        path = root/name
        if path.exists():
            require(path.read_bytes()==payload, 'existing provider receipt differs: '+name)
        else:
            commit(path,payload)
        state[name] = pin(path)
    write(root/'PROVIDER-STARTUP.json',dict(report, local_receipts=state,
                                          remote_receipt_sha256=expected))


if __name__=='__main__':
    main()
