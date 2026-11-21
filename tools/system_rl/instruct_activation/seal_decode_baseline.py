"""Freeze the exact measured release, its declared parent and audited evidence."""
import argparse
import json
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from decode_lineage import digest, read, snapshot, verify


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def seal(config, directory):
    version = config['version']
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', version):
        raise ValueError('invalid version')
    output = directory / version
    if output.exists():
        raise FileExistsError(output)
    packet = Path(config['packet'])
    inventory = {str(f.relative_to(packet)): digest(f) for f in packet.rglob('*') if f.is_file()}
    if inventory != read(Path(config['deployed_manifest'])):
        raise ValueError('custody copy differs from measured deployment')
    profile = read(Path(config['profile']))
    old_lock = Path(profile['decode_parent']['lock'])
    parent = verify(packet, old_lock, read(Path(profile['decode_parent']['declaration'])))
    if read(packet/'DECODE-PARENT.json') != parent or profile['release_kind'] != 'formal':
        raise ValueError('expected formal packet built from declared parent')
    runtime = snapshot(packet)
    evidence = {}
    for label, item in config['windows'].items():
        window = read(Path(item['audit']))
        actual_packet = Path(item['packet'])
        lane, = actual_packet.glob('lane-*/config.json')
        deployed = read(lane)
        deployment = read(Path(item['profile']))
        if (snapshot(actual_packet) != runtime or window['source_boot'] != deployed['host_boot_id']
                or window['candidate_container'] != deployed['name'] + '-' + deployment['attempt']
                or Path(deployed['root']) != Path(deployment['remote_root']) / lane.parent.name):
            raise ValueError('performance measurements belong to a different runtime')
        if (window['reference_ms'] != config['reference_ms'] or window['limit_factor'] != config['factor']
                or not window['all_rounds_within_target'] or len(window['rounds']) != item['rounds']
                or not all(row['concurrency'] == item['concurrency'] and row['requests'] == item['requests']
                           and 0 < row['request_tpot_ms'] <= config['reference_ms'] * config['factor']
                           for row in window['rounds'])):
            raise ValueError('insufficient performance evidence: ' + label)
        # Public evidence uses content hashes, not host names or deployment identifiers.
        evidence[label] = {k:v for k,v in window.items() if k not in ('source_boot','candidate_container')}
    if not evidence:
        raise ValueError('missing performance evidence')
    activation = read(Path(config['activation']))
    policy = read(packet/'bundle/checkpoint/candidate/POLICY.json')
    if (policy['version'] != 'checkpoint-sha256:' + activation['adapter_sha256'] or
            activation['worker_tensors'] != config['worker_tensors'] or
            not all(activation['cache'][k] is True for k in ('kv_cache','prefix_cache','reset_succeeded'))):
        raise ValueError('mismatched adapter or cache evidence')
    action = read(Path(config['actions']))
    if (action['pairs'] != config['expected_pairs'] or action['actions'] <= 0 or
            not all(math.isfinite(action[k]) and action[k]>=0 for k in ('sequence','token')) or
            max(action['token'], action['sequence']) > config['max_logprob_delta'] or action['flips'] != 0):
        raise ValueError('fixed-action acceptance failed')
    settings = {key:profile[key] for key in read(old_lock)['build_settings']}
    directory.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=directory) as tmp:
        stage = Path(tmp) / version
        stage.mkdir()
        write_json(stage/'packet-files.json', inventory)
        write_json(stage/'LOCK.json', dict(version=version, previous_version=parent['parent'],
            previous_lock_sha256=parent['parent_sha256'], build_settings=settings, runtime=runtime,
            packet_files_sha256=digest(stage/'packet-files.json'), measured=evidence,
            activation=activation, fixed_actions=action, training_admission=False))
        write_json(stage/'unchanged.json', dict(parent_sha256=digest(stage/'LOCK.json'),
                                               changes={'engine':{},'files':{},'rows':{}}))
        verify(packet, stage/'LOCK.json',read(stage/'unchanged.json'))
        stage.rename(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(seal(read(args.config), args.directory))
