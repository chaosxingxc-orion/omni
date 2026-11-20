"""Run a configured sequence of closed wires through existing calibration lanes."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from check_policy import check


def main(path):
    root = path.parent
    cfg = json.loads(path.read_bytes())
    stages = [json.loads((root / name).read_bytes()) for name in cfg['stages']]
    if any(Path(stage['output']).exists() for stage in stages):
        raise ValueError('prior diagnostic requests exist; inspect their terminal receipts')
    deadline = time.monotonic() + cfg['ready_seconds']
    for lane in cfg['lanes']:
        launcher = root / ('lane-' + lane) / 'launch.py'
        while subprocess.run(['python3', '-B', str(launcher), 'ready', '--attempt', cfg['attempt']],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            name = json.loads(launcher.with_name('config.json').read_bytes())['name']+'-'+cfg['attempt']
            live = subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',name],text=True)
            if live.strip() != 'true' or time.monotonic() >= deadline:
                raise RuntimeError('calibration service not ready: '+lane)
            time.sleep(5)
    for stage in stages:
        check(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(parser.parse_args().config.resolve())
