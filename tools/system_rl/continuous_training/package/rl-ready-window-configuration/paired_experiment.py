"""Shared, transportable configuration consumed before training or inference starts."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
READERS = {'policy_identity.py', 'publication.py', 'initial_policy.py', 'rank_profile.py'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def validate(exp):
    from rank_profile import validate as validate_rank
    atol = exp.get('publication_norm_atol', 0.0)
    need(type(atol) in (int, float) and math.isfinite(atol) and atol >= 0, 'invalid publication norm tolerance')
    need(exp['schema'] == 'omni-paired-experiment-v1' and exp['version'], 'invalid experiment version')
    need(Path(exp['model_path']).is_absolute() and exp['model'], 'missing model identity')
    devices = exp['training_devices']
    lanes = exp['lanes']
    for group in (devices, *lanes.values()):
        need(isinstance(group, list) and group and all(type(i) is int and i >= 0 for i in group)
             and len(set(group)) == len(group), 'invalid devices')
    physical = [i for group in lanes.values() for i in group]
    need(lanes and len(physical) == len(set(physical)), 'overlapping service devices')
    batch = exp['actor_batch']
    need(set(batch) == {'microbatch_per_card', 'gradient_accumulation_steps', 'arithmetic'}
         and batch['arithmetic'] == 'single_clip_compact_qo_v1', 'unsupported actor batch')
    micro, accum = batch['microbatch_per_card'], batch['gradient_accumulation_steps']
    g, concurrency = exp['episodes_per_recording'], exp['episode_concurrency']
    widths = exp['graph_widths']
    need(all(type(n) is int and n > 0 for n in (micro, accum, g, concurrency, exp['training_memory_gib'])),
         'batch, concurrency and memory must be positive integers')
    need(exp['training_memory_gib'] > 1024 and exp['dev_cadence'] == 'epoch_end', 'invalid training memory or DEV cadence')
    need(isinstance(widths, list) and widths and all(type(n) is int and 1 <= n <= 128 for n in widths)
         and widths == sorted(set(widths)) and widths[0] == 1, 'unsupported decode graph widths')
    total = len(devices) * micro * accum
    need(total % g == 0 and concurrency <= total and concurrency <= widths[-1], 'actor groups or inference capacity differ')
    need(exp.get('row_encoding','original') in ('original','compact_json_v1'), 'unsupported training row encoding')
    need(exp.get('group_preparation','replicated') in ('replicated','shard_by_draw_v1'), 'unsupported group preparation')
    if 'first_wave_forward' in exp:
        need(exp['first_wave_forward']=='overlap_publish_v1' and exp.get('group_preparation')=='shard_by_draw_v1',
             'overlapped first wave requires sharded recording groups')
    need(exp.get('transfer_policy','legacy') in ('legacy','batch_receipt_v1'), 'unknown transfer policy')
    if 'capacity_interval_bytes' in exp:
        need(type(exp['capacity_interval_bytes']) is int and exp['capacity_interval_bytes'] > 0,
             'invalid checkpoint capacity interval')
    width = exp.get('activation_concurrency', 1)
    need(type(width) is int and 1 <= width <= len(lanes), 'invalid activation concurrency')
    validate_rank(rank_profile(exp))
    need(set(exp['readers']) == READERS and all(re.fullmatch('[a-f0-9]{64}', h) for h in exp['readers'].values()),
         'incomplete paired reader lock')
    return total // g


def rank_profile(exp):
    return dict(schema='p-lora-profile-v1', training={k: exp['lora'][k] for k in ('lora_rank','lora_alpha','lora_dropout')},
                inference={'max_lora_rank': exp['lora']['lora_rank']}, actor_batch={k: exp['actor_batch'][k] for k in ('microbatch_per_card','gradient_accumulation_steps','arithmetic')},
                dev_cadence=exp['dev_cadence'])


def graph(exp):
    widths = exp['graph_widths']
    return dict(schema='tp4-decode-graph-widths-v1', batch_sizes=widths, tiled_attention_batch=widths[-1],
                max_num_seqs=widths[-1])


def load(path, expected=None):
    exp = json.loads(Path(path).read_bytes())
    validate(exp)
    if expected is not None:
        need(digest(exp) == expected, 'different experiment snapshot')
    return exp


def check_readers(exp, installed):
    for name, expected in exp['readers'].items():
        need(installed.get(name) == expected, 'different paired adapter reader: ' + name)


def check_actor(exp, start, policy):
    training = start['training']
    need(start['devices'] == exp['training_devices'] and training['world_size'] == len(exp['training_devices'])
         and start['actor_batch'] == exp['actor_batch'] and training['model_path'] == exp['model_path']
         and all(training[k] == v for k, v in exp['lora'].items()), 'actor differs from experiment')
    need(policy['experiment_sha256'] == digest(exp) and policy['dev_cadence'] == exp['dev_cadence']
         and policy['rollout_dispatch']['episode_concurrency'] == exp['episode_concurrency']
         and all(policy.get(key) == exp.get(key) for key in
                 ('row_encoding', 'group_preparation', 'first_wave_forward', 'activation_concurrency', 'capacity_interval_bytes', 'transfer_policy'))
         and policy['acceptance']['episodes_per_recording'] == exp['episodes_per_recording']
         and policy['acceptance']['recordings_per_step'] == validate(exp), 'controller differs from experiment')


def check_lane(exp, lane, cfg):
    args = cfg['serve_args']
    need(cfg['served_model_name'] == exp['model'] and cfg['model'] == exp['model_path']
         and cfg['cards'] == exp['lanes'][lane] and args.count('--tensor-parallel-size') == 1
         and int(args[args.index('--tensor-parallel-size') + 1]) == len(cfg['cards']),
         'service model or topology differs from experiment')


def check_effective_lane(exp, lane, cfg, adapter):
    check_lane(exp, lane, cfg)
    expected = rank_profile(exp)
    fields = {'r': 'lora_rank', 'lora_alpha': 'lora_alpha', 'lora_dropout': 'lora_dropout'}
    for key, setting in fields.items():
        need(adapter.get(key) == expected['training'][setting],
             f'published adapter {key} differs from experiment')
    args = cfg['serve_args']
    limits = {'--max-lora-rank': expected['inference']['max_lora_rank'],
              '--max-num-seqs': graph(exp)['max_num_seqs']}
    for flag, value in limits.items():
        matches = [i for i, arg in enumerate(args) if arg == flag or arg.startswith(flag + '=')]
        need(len(matches) == 1, f'lane {lane} requires one {flag}')
        i = matches[0]
        actual = args[i+1:i+2] if args[i] == flag else [args[i].split('=', 1)[1]]
        need(actual == [str(value)], f'lane {lane} {flag} differs from experiment')


def install_projection(path, value):
    """Same bytes as build.save, including key order, on either host."""
    import tempfile
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    with tempfile.NamedTemporaryFile(dir=path.parent) as temp:
        temp.write(raw)
        temp.flush()
        os.fsync(temp.fileno())
        try:
            os.link(temp.name, path)
        except FileExistsError:
            need(path.read_bytes() == raw, 'immutable projection changed')
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def compile_experiment(template, profile, output):
    from build import save, pin
    from source.config_profile import compile_profile
    receipt = compile_profile(template, profile, output / 'resolution')
    resolution = json.loads(Path(receipt['path']).read_bytes())
    exp = json.loads(Path(resolution['resolved_ref']['path']).read_bytes())
    base_name = exp.pop('policy_base')
    need(base_name == Path(base_name).name, 'expected a packaged policy')
    parent = HERE / base_name
    policy = json.loads(parent.read_bytes())
    exp['policy_base_sha256'] = file_hash(parent)
    exp['readers'] = {}
    roster = {p for members in policy['source_roles'].values() for p in members}
    for name, member in exp.pop('reader_sources').items():
        need(member in roster and Path(member).name == name, 'reader outside policy source roster')
        exp['readers'][name] = file_hash(HERE.parent / member)
    recordings = validate(exp)
    snapshot = save(output / 'EXPERIMENT.json', exp)
    rank = save(output / 'rank-profile.json', rank_profile(exp))
    policy.update(version=exp['version'], experiment_sha256=digest(exp), dev_cadence=exp['dev_cadence'])
    for name in ('row_encoding', 'group_preparation', 'first_wave_forward', 'activation_concurrency', 'capacity_interval_bytes', 'transfer_policy'):
        if name in exp:
            policy[name] = exp[name]
    policy['rollout_dispatch']['episode_concurrency'] = exp['episode_concurrency']
    policy['acceptance'].update(recordings_per_step=recordings, episodes_per_recording=exp['episodes_per_recording'])
    target_reader='rl-ready-window-configuration/training_target.py'
    policy['source_roles']['controller'].append(target_reader)
    policy['runtime_members']['parents/jev-system-rl/'+target_reader]=target_reader
    historical_graph = policy.pop('decode_graph_profile')
    policy['source_roles']['service'].remove(historical_graph)
    policy['runtime_members'].pop('parents/jev-system-rl/' + historical_graph)
    if 'capacity_interval_bytes' in exp:
        source = 'rl-ready-window-configuration/source/checkpoint_inventory.py'
        policy['source_roles']['controller'].append(source)
        policy['runtime_members']['parents/jev-system-rl/implementation/checkpoint_inventory.py'] = source
    if exp.get('transfer_policy') == 'batch_receipt_v1':
        source = 'rl-ready-window-configuration/retire_input_transfer.py'
        policy['source_roles']['controller'].append(source)
        policy['runtime_members']['parents/jev-system-rl/implementation/retire_input_transfer.py'] = source
    if exp.get('first_wave_forward'):
        for filename in ('first_wave_forward.py','microbatch_producer.py','microbatch_fields.py','microbatch_roster.py','policy_row_assignment.py'):
            member='rl-ready-window-configuration/source/'+filename
            if member not in policy['source_roles']['actor']:policy['source_roles']['actor'].append(member)
            policy['runtime_members']['parents/jev-system-rl/'+member]=member
    scoring = 'implementation/scorer_extension.py'
    policy['source_roles']['shared'].append(scoring)
    policy['runtime_members']['parents/jev-system-rl/' + scoring] = scoring
    for name in ('paired_experiment.py', 'configure_branch_training.py', 'deploy_resident_graph_lanes.py',
                 'P-EXPERIMENT.template.json', 'EXPERIMENT-DEPLOYMENT.example.json', base_name,
                 'BRANCH-TRAINING.template.json', 'BRANCH-TRAINING-DEPLOYMENT.example.json'):
        member = 'rl-ready-window-configuration/' + name
        policy['source_roles']['controller'].append(member)
        policy['runtime_members']['parents/jev-system-rl/' + member] = member
    return save(output / 'PACKAGE.json', dict(experiment_ref=snapshot, experiment_sha256=digest(exp),
                resolution_ref=receipt, rank_profile_ref=rank, graph_ref=save(output / 'graph.json', graph(exp)),
                policy_ref=save(output / 'TRAINING-POLICY.json', policy), policy_base_ref=pin(parent)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=HERE / 'P-EXPERIMENT.template.json')
    parser.add_argument('--profile', type=Path, default=os.getenv('OMNI_DEPLOYMENT_PROFILE') or HERE / 'EXPERIMENT-DEPLOYMENT.example.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--release', type=Path, default=os.getenv('OMNI_TRAIN_RELEASE') or None)
    args = parser.parse_args()
    if args.release:
        import sys
        sys.path.insert(0, str(args.release / 'tree/parents/jev-system-rl/implementation'))
    print(json.dumps(compile_experiment(args.template, args.profile, args.output)))
