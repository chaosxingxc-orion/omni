"""Run the registered paired full1080 read once after A0 official scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from real500_successor_queue import D, U, IMAGE, write, status


def ready(parent, complete):
    if parent.get('state') == 'failed': raise ValueError('A0 source read failed; preserve evidence')
    if parent.get('state') != 'complete': return False
    if complete is None or complete.get('rows') != 1080:
        raise ValueError('full1080 completion missing')
    return True


def run(out, source_queue=None):
    if out.exists() or not out.resolve().is_relative_to(D / 'work/codex'):
        raise ValueError('fresh owned queue required')
    study = U / 'studies/speech-aware-evidence-acquisition'
    source = D / 'analysis/SAEA-SLURP-A0-FULL1080-SOURCE-READ-V1'
    target = D / 'analysis/SAEA-A0-DIRECT-FULL1080-COMPARISON-V1.json'
    if target.exists(): raise ValueError('comparison already exists')
    paths = [study / 'scripts/analysis/compare_a0_direct_full1080.py',
        study / 'docs/readiness/2026-10-31-a0-direct-full1080-comparison/analysis-recipe-v1.json']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    pins = {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    write(out / 'intent.json', {'pid': os.getpid(), 'pins': pins, 'interval_seconds': 600,
        'new_model_calls': 0, 'new_scoring_jobs': 0, 'automatic_retry': False,
        'source_queue': str(source_queue or D/'work/codex/a0-full1080-result-queue-v1')})
    source_queue = source_queue or D / 'work/codex/a0-full1080-result-queue-v1'
    if not source_queue.resolve().is_relative_to(D/'work/codex'): raise ValueError('owned source queue required')
    read = lambda p: json.loads(p.read_bytes()) if p.exists() else None
    status(out, 'waiting_full1080_official_score')
    try:
        while not ready(read(source_queue / 'status.json') or {}, read(source / 'complete.json')):
            time.sleep(600)
        if any(sha(Path(p)) != digest for p, digest in pins.items()): raise ValueError('comparison source changed')
        status(out, 'verifying_and_comparing_full1080')
        env = {**os.environ, 'TREE': str(U), 'CPUSET': '176-191', 'PORT_CONTAINER_USER': '0:0',
            'PORT_IMAGE': IMAGE, 'PORT_DOCKER_FLAGS': '--network none --name saea-a0-direct-full1080-comparison-v1'}
        args = ['bash', str(D / 'work/scripts/port_py.sh'), 'scripts/analysis/compare_a0_direct_full1080.py',
            '--data', str(D), '--parent', str(D / 'work/dev/real500-runtime/studies/speech-aware-evidence-acquisition/configs/dataset/slurp-speech-native-evo-fold-b-1080.json'),
            '--a0', str(source), '--out', str(target)]
        with (out / 'read.log').open('x') as log:
            result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(out / 'formal.exit.json', {'exit': result.returncode, 'time': time.time()})
        if result.returncode: raise ValueError('paired read failed; do not retry automatically')
        comparison = read(target)
        if comparison['rows'] != 1080 or not comparison['complete_comparison']:
            raise ValueError('incomplete paired comparison')
        status(out, 'complete', rows=1080, comparison_sha256=sha(target),
            comparisons=comparison['comparisons'], next_required='Review full evidence and lock first-system baseline before TFRL.')
    except BaseException as error:
        status(out, 'failed', error=str(error), automatic_retry=False)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-queue',type=Path)
    run(**vars(p.parse_args()))
