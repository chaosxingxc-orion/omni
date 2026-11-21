"""External service supervision: no changes to vLLM/image; local durable receipts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
HOT = Path('/runtime')
GIB = 1024**3


def durable(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fs(path):
    path = Path(path).resolve(strict=True)
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        b, a = before.split(), after.split()
        mp = Path(b[4].replace('\\040', ' '))
        if path == mp or mp in path.parents:
            mounts.append((len(str(mp)), a[0], b[5].split(','), str(mp)))
    _, kind, flags, mp = max(mounts)
    assert kind in ('ext4', 'xfs', 'btrfs') and 'rw' in flags, (path, kind, flags)
    v = os.statvfs(path)
    free, total = v.f_bavail * v.f_frsize, v.f_blocks * v.f_frsize
    assert free >= 20*GIB and free >= total*.10, (path, free, total)
    assert v.f_favail >= v.f_files*.05, (path, 'inodes')
    return dict(path=str(path), mount=mp, fstype=kind, free_bytes=free,
                total_bytes=total, free_inode_fraction=v.f_favail/v.f_files)


def usage():
    # Count allocated blocks once per inode (no cache deletion, no shared writes).
    seen, blocks = set(), 0
    for parent, dirs, files in os.walk(HOT):
        for name in dirs + files:
            st = (Path(parent)/name).lstat()
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                seen.add(key)
                blocks += st.st_blocks * 512
    return blocks


def check():
    paths = [HOT, Path(os.environ['HOME'])/'ascend/log', Path('/tmp'), Path('/jev-tmp')]
    keys = ['TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCHINDUCTOR_CACHE_DIR',
            'TRITON_CACHE_DIR', 'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH',
            'ASCEND_PROCESS_LOG_PATH', 'VLLM_CACHE_ROOT', 'TORCH_HOME', 'VLLM_RPC_BASE_PATH']
    paths += [Path(os.environ[k]) for k in keys]
    facts = [fs(p) for p in paths]
    used = usage()
    assert used < CFG['runtime_budget_bytes'], ('runtime budget exhausted', used)
    # Keep remaining declared budget in addition to minimum reservation.
    remaining = CFG['runtime_budget_bytes'] - used
    for f in facts:
        assert f['free_bytes'] >= max(20*GIB, f['total_bytes']*.10) + remaining + 300*1024**2, f
    assert len((os.environ['VLLM_RPC_BASE_PATH'] + '/' + 'x'*36).encode()) <= 107
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == ','.join(map(str,range(len(CFG['cards']))))
    assert os.environ['HCCL_SOCKET_IFNAME'] == CFG['socket_ifname']
    model = Path(CFG['model'])
    config = json.loads((model/'config.json').read_text())
    assert config['architectures'] == ['Qwen3OmniMoeForConditionalGeneration']
    index = json.loads((model/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    assert len(shards) == 15 and all((model/s).stat().st_size > 0 for s in shards)
    return dict(time=time.time(), storage=facts, runtime_allocated_bytes=used,
                model_shards=len(shards), config_sha256=hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest())


def offline():
    result = check()
    # Exercise short AF_UNIX IPC locally; no Ascend/torch/vLLM import.
    p = Path('/jev-tmp')/('preflight-' + str(os.getpid()) + '.sock')
    with socket.socket(socket.AF_UNIX) as s:
        try:
            s.bind(str(p))
        finally:
            p.unlink(missing_ok=True)
    if 'policy_overlay' in CFG:
        subprocess.run(service_argv(check_only=True), env=child_environment(), check=True)
    # Exercise real local file + directory synchronization, retain receipt.
    durable(HOT/'state'/'PREFLIGHT.json', result)
    print(json.dumps(result, indent=2))


def service_argv(*,check_only=False):
    python=sys.executable
    common=['--served-model-name', CFG['served_model_name'], '--host', CFG['host'], '--port', str(CFG['port'])]
    if 'policy_overlay' in CFG:
        overlay=CFG['policy_overlay']
        return [python,'-B',str(Path(overlay['bundle'])/'parents/jev-system-rl/implementation/service_bootstrap.py'),
                '--config',overlay['server'],*(['--check-only'] if check_only else []),
                '--','--model',CFG['model'],*common,*CFG['serve_args']]
    return [python,'-m','vllm.entrypoints.cli.main','serve',CFG['model'],*common,*CFG['serve_args']]


def child_environment():
    env=os.environ.copy()
    if 'policy_overlay' in CFG:
        bundle=Path(CFG['policy_overlay']['bundle'])
        env['PYTHONPATH']=os.pathsep.join(filter(None,(
            str(bundle/'parents/policy-iteration-successor'),env.get('PYTHONPATH'))))
    return env


def serve():
    state = HOT/'state'
    guard = (state/'service.guard').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = check()
    argv=service_argv()
    # Persist launch intent before any engine creation. No silent restart/retry.
    launch = dict(result, argv=argv, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    durable(state/'LAUNCH.json', launch)
    child = subprocess.Popen(argv, env=child_environment(), start_new_session=True)
    stopped = []
    def stop(sig, frame):
        stopped.append('signal:' + str(sig))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    next_check = time.monotonic() + CFG['monitor_seconds']
    while child.poll() is None:
        if stopped:
            break
        if time.monotonic() >= next_check:
            try:
                report = check()
                durable(state/'STORAGE.json', report)
            except Exception as e:
                stopped.append('storage:' + repr(e))
                break
            next_check = time.monotonic() + CFG['monitor_seconds']
        time.sleep(1)
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=CFG['stop_timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    terminal = dict(time=time.time(), exit_code=child.returncode, stop_reasons=stopped)
    durable(state/('STORAGE-STOP.json' if any(x.startswith('storage:') for x in stopped) else 'EXIT.json'), terminal)
    print(json.dumps(terminal), flush=True)
    return child.returncode if child.returncode >= 0 else 128-child.returncode


if __name__ == '__main__':
    if sys.argv[1:] == ['check']:
        offline()
    elif sys.argv[1:] == ['serve']:
        sys.exit(serve())
    else:
        raise SystemExit('usage: service.py check|serve')
