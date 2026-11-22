"""Host launcher. MAIN executes start; check is CPU-only and maps no devices."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import urllib.request

CODE = Path(__file__).resolve().parent
C = json.loads((CODE/'config.json').read_text())
LABEL = 'omni.activation=' + C['name']


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def verify():
    pins = json.loads((CODE/'PINS.json').read_text())
    for name, digest in pins.items():
        assert hashlib.sha256((CODE/name).read_bytes()).hexdigest() == digest, name
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == C['host_boot_id'], 'boot changed; recheck resource ownership'
    assert command(['docker', 'image', 'inspect', C['image'], '--format', '{{.Id}}']) == C['image']
    assert command(['findmnt', '-T', C['root'], '-n', '-o', 'FSTYPE']) in ('ext4', 'xfs', 'btrfs')
    assert C['cards'] in ([0, 1, 2, 3], [4, 5, 6, 7])
    assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,45}', C['name'])
    assert Path(C['guard']).is_file(), 'configured host guard required'
    assert command(['findmnt', '-T', C['guard'], '-n', '-o', 'FSTYPE']) in ('ext4', 'xfs', 'btrfs')
    assert Path(C['model']).resolve().is_relative_to(Path(C['readonly_root']).resolve())
    assert C['host'] == '127.0.0.1'
    assert all(type(C[k]) is int and 0 < C[k] < 65536 for k in ('port','vllm_port','hccl_if_base_port'))
    reserved = range(C['hccl_if_base_port'], C['hccl_if_base_port'] + C['hccl_port_reservation'])
    assert C['port'] != C['vllm_port'] and C['port'] not in reserved and C['vllm_port'] not in reserved and reserved.stop <= 65536
    assert C['serve_args'].count('--enable-prefix-caching') == 1 and '--no-enable-prefix-caching' not in C['serve_args']
    assert C['client_chat_template_kwargs'] == {'enable_thinking': False}
    assert C['serve_args'][C['serve_args'].index('--tensor-parallel-size')+1] == str(len(C['cards']))
    if 'policy_overlay' in C:
        overlay=C['policy_overlay']
        assert set(overlay)-{'successor_mount','update_mount'}=={'bundle','server','checkpoint_mount'} and Path(overlay['bundle']).is_dir()
        assert Path(overlay['server']).is_relative_to(Path(overlay['bundle']))
        assert command(['findmnt','-T',overlay['bundle'],'-n','-o','FSTYPE']) in ('ext4','xfs','btrfs')
        if 'update_mount' in overlay:
            inbox=Path(overlay['update_mount'])
            assert inbox.is_absolute() and inbox.resolve()==inbox and inbox.is_dir()
            assert command(['findmnt','-T',str(inbox),'-n','-o','FSTYPE']) in ('ext4','xfs','btrfs')


def name(attempt):
    assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,25}', attempt), 'invalid attempt'
    return C['name'] + '-' + attempt


def inspect(n):
    d = json.loads(command(['docker', 'inspect', n]))[0]
    assert d['Image'] == C['image']
    assert d['Config']['Labels'].get('omni.activation') == C['name']
    return d


def argv(attempt, start):
    run = Path(C['root'])/'runs'/attempt
    run.parent.mkdir(mode=0o750, exist_ok=True)
    run.mkdir(mode=0o750, exist_ok=False)
    for d in ['home/ascend/log', 'cache/huggingface', 'cache/inductor', 'cache/triton',
              'cache/ascend', 'cache/vllm', 'cache/torch', 'cache/pycache', 'tmp', 'work', 'logs', 'state']:
        (run/d).mkdir(parents=True, exist_ok=True, mode=0o750)
    env = dict(HOME='/runtime/home', TMPDIR='/jev-tmp', TMP='/jev-tmp', TEMP='/jev-tmp',
               VLLM_RPC_BASE_PATH='/jev-tmp', XDG_CACHE_HOME='/runtime/cache',
               HF_HOME='/runtime/cache/huggingface', TORCHINDUCTOR_CACHE_DIR='/runtime/cache/inductor',
               TRITON_CACHE_DIR='/runtime/cache/triton', ASCEND_WORK_PATH='/runtime/work',
               ASCEND_CACHE_PATH='/runtime/cache/ascend', ASCEND_PROCESS_LOG_PATH='/runtime/home/ascend/log',
               VLLM_CACHE_ROOT='/runtime/cache/vllm', TORCH_HOME='/runtime/cache/torch',
               PYTHONPYCACHEPREFIX='/runtime/cache/pycache', PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               # Docker exposes only the physical device nodes below. CANN
               # numbers the visible set locally, including for cards 4–7.
               ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,range(len(C['cards'])))),
               ASCEND_VISIBLE_DEVICES=','.join(map(str,C['cards'])),
               HCCL_IF_BASE_PORT=str(C['hccl_if_base_port']), HCCL_SOCKET_IFNAME=C['socket_ifname'],
               GLOO_SOCKET_IFNAME=C['socket_ifname'], TP_SOCKET_IFNAME=C['socket_ifname'],
               VLLM_PORT=str(C['vllm_port']))
    if 'batch_invariant' in C:
        assert type(C['batch_invariant']) is bool, 'batch_invariant must be a boolean'
        env['VLLM_BATCH_INVARIANT'] = '1' if C['batch_invariant'] else '0'
    a = ['docker', 'run', '-d' if start else '--attach=stdout', '--name', name(attempt),
         '--label', LABEL, '--network', 'host' if start else 'none', '--ipc', 'private',
         '--shm-size', '16g', '--read-only', '--restart', 'no',
         '--security-opt', 'no-new-privileges', '--log-driver', 'json-file',
         '--log-opt', 'max-size=100m', '--log-opt', 'max-file=3',
         '--stop-timeout', str(C['stop_timeout_seconds']+15), '--workdir', '/runtime']
    for k, v in env.items():
        a += ['-e', k+'='+v]
    mounts = [(str(run), '/runtime', False), (str(run/'tmp'), '/tmp', False),
              (str(run/'tmp'), '/jev-tmp', False), (str(CODE), '/activation', True),
              (C['readonly_root'], C['readonly_root'], True)]
    if 'policy_overlay' in C:
        bundle=C['policy_overlay']['bundle']
        if 'successor_mount' in C['policy_overlay']:
            mounts.append((bundle+'/successor',C['policy_overlay']['successor_mount'],True))
        if 'update_mount' in C['policy_overlay']:
            path=C['policy_overlay']['update_mount']
            mounts.append((path,path,True))
        mounts.extend([(bundle,bundle,True),(bundle+'/checkpoint',C['policy_overlay']['checkpoint_mount'],True),
                       (C['guard'],C['guard'],True)])
    for p in ['/usr/local/Ascend/driver', '/usr/local/sbin/npu-smi', '/etc/hccn.conf',
              '/etc/ascend_install.info', '/usr/local/Ascend/firmware', '/usr/local/dcmi']:
        assert Path(p).exists(), p
        mounts.append((p, p, True))
    if start:
        for p in ['/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'] + ['/dev/davinci'+str(i) for i in C['cards']]:
            a += ['--device', p]
    for src, dst, ro in mounts:
        a += ['--mount', 'type=bind,src='+src+',dst='+dst+(',readonly' if ro else '')]
    return a + ['--entrypoint', '/usr/local/python3.12.13/bin/python3', C['image'],
                '-B', '/activation/service.py', 'serve' if start else 'check']


def start_checks():
    # No NPU inference: reject an occupied port or any active process on authorized cards.
    for p in [C['port'], C['vllm_port']] + list(range(C['hccl_if_base_port'], C['hccl_if_base_port']+C['hccl_port_reservation'])):
        with socket.socket() as s:
            # Match normal server bind semantics; TIME_WAIT is not a listener.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((C['host'] if p == C['port'] else '0.0.0.0', p))
    for card in C['cards']:
        out = command(['npu-smi', 'info', '-t', 'proc-mem', '-i', str(card)])
        assert 'No process in device.' in out, ('busy/unreadable card', card)
    for cid in command(['docker', 'ps', '-q']).split():
        d = json.loads(command(['docker', 'inspect', cid]))[0]
        # Do not change other containers. Detect a conflicting explicit device mapping.
        ds = d['HostConfig'].get('Devices') or []
        assert not any(x['PathOnHost'] in ['/dev/davinci'+str(i) for i in C['cards']] for x in ds), ('device lease conflict', d['Name'])


def ready(attempt, expected_adapter=None):
    d = inspect(name(attempt))
    assert d['State']['Running'], d['State']
    assert sorted(x['PathOnHost'] for x in d['HostConfig']['Devices']) == sorted(
        ['/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc']+['/dev/davinci'+str(i) for i in C['cards']])
    assert not d['HostConfig']['Privileged'] and d['HostConfig']['ReadonlyRootfs']
    base = 'http://'+C['host']+':'+str(C['port'])
    with urllib.request.urlopen(base+'/health', timeout=8) as r:
        assert r.status == 200
    with urllib.request.urlopen(base+'/v1/models', timeout=8) as r:
        models = json.load(r)['data']
    model = C['served_model_name']
    bases = [m for m in models if m['id'] == model]
    adapters = [m for m in models if m['id'] != model]
    assert len(bases) == 1 and bases[0].get('max_model_len') == int(C['serve_args'][C['serve_args'].index('--max-model-len')+1]), models
    if adapters:
        assert 'policy_overlay' in C and len(adapters) == 1, models
        adapter, = adapters
        assert re.fullmatch(r'checkpoint-sha256:[0-9a-f]{64}', adapter['id']) and adapter.get('parent') == model, models
    if expected_adapter is not None:
        assert len(adapters) == 1 and adapters[0]['id'] == expected_adapter, 'active adapter mismatch'
        receipt = json.loads(command(['docker', 'exec', d['Id'], 'cat', '/runtime/state/policy/ACTIVATION.json']))
        assert receipt['alias'] == receipt['policy_version'] == expected_adapter
        assert 'checkpoint-sha256:'+receipt['weights_ref']['sha256'] == expected_adapter
        assert receipt['service_identity']['container_name'] == name(attempt) and receipt['source_boot'] == C['host_boot_id']
        assert receipt['base'] is False and receipt['fixture'] is False
    print(json.dumps(dict(ready=True, container_id=d['Id'], models=models, thinking='OFF; clients send chat_template_kwargs.enable_thinking=false')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['check', 'start', 'ready', 'stop', 'inspect'])
    p.add_argument('--attempt', default='run01')
    p.add_argument('--expected-adapter')
    p.add_argument('--preflight-attempt', default='check01')
    ns = p.parse_args()
    verify()
    # Serialize this project's local launch attempts; never replace an existing lock.
    launch_guard = Path(C['guard']).open('a+')
    fcntl.flock(launch_guard, fcntl.LOCK_EX)
    n = name(ns.attempt)
    if ns.action in ('check', 'start'):
        if ns.action == 'start':
            name(ns.preflight_attempt)
            receipt = Path(C['root'])/'runs'/ns.preflight_attempt/'state'/'PREFLIGHT.json'
            assert receipt.is_file(), 'successful device-free check required'
            assert json.loads(receipt.read_text())['config_sha256'] == hashlib.sha256((CODE/'config.json').read_bytes()).hexdigest()
            start_checks()
        a = argv(ns.attempt, ns.action == 'start')
        print(command(a), flush=True)
    elif ns.action == 'ready':
        ready(ns.attempt, ns.expected_adapter)
    elif ns.action == 'inspect':
        d = inspect(n)
        print(json.dumps({k:d[k] for k in ('Id', 'Image', 'State', 'Mounts')}, indent=2))
    else:
        d = inspect(n)
        assert d['State']['Running'], 'already stopped; preserve evidence'
        print(command(['docker', 'stop', '-t', str(C['stop_timeout_seconds']+15), d['Id']]))
