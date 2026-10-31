"""Wait for the existing CPU encoder, then derive inputs once without models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

D = Path('/mnt/cephfs/xingchao/omni')
IMAGE = 'sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
CONFIG_SHA = 'f665688b11f9ff74667b6ed546445e79fa963fa4008e8fb94f6474f5e1ef6c2a'


def run(tree, expected_head, out):
    if not out.resolve().is_relative_to(D / 'work/codex'):
        raise ValueError('owned queue directory required')
    out.mkdir(exist_ok=False)
    def status(state, **facts):
        temp = out / 'status.tmp'
        temp.write_text(json.dumps({'state': state, 'pid': os.getpid(),
            'observed_at': time.time(), 'interval_seconds': 600, **facts}, indent=2) + '\n')
        temp.replace(out / 'status.json')
    terminal = D / 'work/codex/host-recovery-client-v1/r01c-cpu-encoding-v1.exit'
    status('waiting_encoder_terminal')
    try:
        while not terminal.exists():
            time.sleep(600)
        if terminal.read_text().strip() != '0':
            raise ValueError('encoder failed; preserve source and do not replay')
        study = tree / 'studies/speech-aware-evidence-acquisition'
        git = ['git', '-c', 'safe.directory=' + str(study), '-C', str(study)]
        if (subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() != expected_head
                or subprocess.check_output(git + ['status', '--porcelain']).strip()):
            raise ValueError('derivative source changed')
        status('deriving', source_commit=expected_head)
        env = {**os.environ, 'TREE': str(tree), 'CPUSET': '160-175',
            'PORT_CONTAINER_USER': '0:0', 'PORT_IMAGE': IMAGE,
            'PORT_DOCKER_FLAGS': '--network none --name saea-r01c-retrieval-inputs-v1'}
        args = ['bash', str(D / 'work/scripts/port_py.sh'),
            'scripts/flights/nx/r01c_retrieval_inputs.py',
            '--encoder-repo', str(D / 'work/dev/r01c-cpu-overlap-v1/studies/speech-aware-evidence-acquisition'),
            '--data', str(D), '--runtime-config', str(D / 'work/codex/r01c-cpu-runtime-config-v1.json'),
            '--runtime-sha', CONFIG_SHA, '--exit-path', str(terminal), '--out', str(out / 'derivative')]
        with (out / 'derivative.log').open('x') as log:
            result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
        (out / 'formal.exit').write_text(str(result.returncode) + '\n')
        if result.returncode:
            raise ValueError('derivative failed; inspect partial artifacts without replay')
        receipt = json.loads((out / 'derivative/complete.json').read_bytes())
        if receipt['rows'] != 625 or not receipt['values_unchanged']:
            raise ValueError('derivative completion differs')
        status('complete', rows=625, new_model_contacts=0, new_encoder_contacts=0,
            completion=str(out / 'derivative/complete.json'), claims_tfrl_admitted=False)
    except BaseException as error:
        status('failed', error=str(error), automatic_retry=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(**vars(parser.parse_args()))
