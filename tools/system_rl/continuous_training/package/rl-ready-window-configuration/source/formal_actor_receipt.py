"""Project a terminal FSDP8 update into the original controller's result schema."""
import hashlib
import math
from pathlib import Path

from dependencies import require
from policy_identity import digest
from run_package import pin, read, write


def reward_diagnostics(report, group_ids):
    groups = {row['group_id']: row for row in report['group_results']}
    require(len(groups) == len(report['group_results']) and set(groups) == set(group_ids),
            'reward report differs from the committed group roster')
    per_group = []
    nonzero = 0
    for group_id in group_ids:
        episodes = groups[group_id]['episodes']
        if not episodes:
            per_group.append(dict(group_id=group_id, episodes=0, mean_reward=None,
                                  population_variance=None))
            continue
        values = [float(row['reward']) for row in episodes]
        advantages = [float(row['advantage']) for row in episodes]
        require(all(math.isfinite(x) for x in values + advantages), 'nonfinite reward observation')
        mean = math.fsum(values) / len(values)
        nonzero += sum(x != 0 for x in advantages)
        per_group.append(dict(group_id=group_id, episodes=len(values), mean_reward=mean,
                              population_variance=math.fsum((x - mean)**2 for x in values) / len(values)))
    denominator = sum(group['episodes'] for group in per_group)
    return dict(schema='sampling-reward-signal-v1', groups=per_group, episodes=denominator,
                nonzero_advantage_episodes=nonzero,
                nonzero_advantage_fraction=nonzero / denominator if denominator else None,
                source='original training consumer group_results', extra_forward=0)


def terminal_ranks(exit_ref, rank_refs):
    terminal = read(exit_ref)
    require(terminal.get('exit_code') == 0 and terminal.get('failure') is None
            and terminal.get('complete_ranks') == 8 and terminal.get('container_id'),
            'training container must exit successfully with all eight ranks closed')
    rows = [read(ref) for ref in rank_refs]
    require(len(rows) == 8 and [row.get('rank') for row in rows] == list(range(8)),
            'invalid rank receipt roster')
    return rows


def accepted_update(exit_ref, rank_refs, prior_checkpoint_ref, checkpoint_ref, report_ref,
                    *, prior_clock=None, generation_ref=None):
    """The writer must close these refs under the original run and boot."""
    return _updated(terminal_ranks(exit_ref,rank_refs),rank_refs,prior_checkpoint_ref,checkpoint_ref,report_ref,
                    prior_clock=prior_clock,generation_ref=generation_ref,terminal={'exit_ref':exit_ref})


def _updated(rows,rank_refs,prior_checkpoint_ref,checkpoint_ref,report_ref,*,prior_clock,generation_ref,terminal):
    before = read(prior_checkpoint_ref)
    after = read(checkpoint_ref)
    step = after['step']
    clock = after['clock']
    old_clock = before['clock'] if prior_clock is None else prior_clock
    lineage = after['training_lineage']
    groups = lineage['group_ids']
    attempt = lineage['attempt_id']
    require(before['world_size'] == after['world_size'] == 8
            and old_clock['successful_optimizer_steps'] == before['step'] == step - 1
            and step == clock['successful_optimizer_steps'], 'checkpoint did not advance exactly one step')
    require(after['recoverable'] is True and after['contract']['lineage'] == before['contract']['lineage'],
            'optimizer lineage changed')
    require(groups and len(groups) == len(set(groups)) and set(groups).isdisjoint(old_clock['groups'])
            and set(clock['groups']) == set(old_clock['groups']) | set(groups)
            and all(clock['groups'][key] == value for key, value in old_clock['groups'].items())
            and all(clock['groups'][key]['consumed'] is True for key in groups)
            and attempt not in old_clock['attempts']
            and clock['attempts'] == {**old_clock['attempts'], attempt: dict(group_ids=groups, succeeded=True)},
            'checkpoint did not commit these fresh groups exactly once')
    publication = after['resident_identity']['publication']['identity_sha256']
    objective = rows[0]['objective']
    require(all(row['step'] == step and row['recoverable_checkpoint'] is True
                and row['adapter_export'] is True and row['objective'] == objective
                and row['publication_sha256'] == publication for row in rows),
            'ranks disagree on the committed objective or adapter')
    require(objective['schema'] == 'owned-global-true-objective-v1'
            and objective['reduction'] == 'sum_ranks_already_global_normalized'
            and objective['padding_included'] is False and objective['optimizer_step'] is True
            and objective['snapshot']['successful_optimizer_steps'] == step - 1
            and objective['snapshot'] == lineage['annealing_snapshot']
            and all(math.isfinite(float(v)) for v in objective['components'].values()),
            'expected finite global actor objective for the committed optimizer step')
    grad_norms = []
    for row in rows:
        raw_norm = row['metrics']['actor/grad_norm']
        if isinstance(raw_norm, list):
            raw_norm, = raw_norm
        grad_norms.append(float(raw_norm))
    norm = grad_norms[0]
    require(math.isfinite(norm) and norm > 0 and all(n == norm for n in grad_norms),
            'ranks disagree on the finite nonzero global gradient norm')
    report = read(report_ref)
    reward = reward_diagnostics(report, groups)
    require(objective['global_episode_denominator'] == report['global_episode_denominator'] > 0
            and report['action_tokens'] > 0, 'objective denominator differs from the accepted consumer')
    observation = dict(objective,
                       snapshot_sha256=objective.get('snapshot_sha256') or objective['snapshot'].get('snapshot_sha256')
                       or digest(objective['snapshot']),
                       successful_step_before=step - 1, successful_step_after=step,
                       sampling_reward_diagnostics=reward, action_tokens=report['action_tokens'],
                       groups=len(groups), gradient_norm=norm)
    return dict(status='updated', optimizer_step=True, optimizer_steps=1, clock=clock,
                checkpoint_ref=checkpoint_ref, generation_inventory_ref=generation_ref,
                group_ids=groups, attempt_id=attempt, rank_refs=rank_refs, **terminal,
                result=dict(optimizer_step=True, objective_observation=observation))


def accepted_zero(exit_ref, rank_refs, prior_checkpoint_ref, prior_clock,
                  *, verify_clock, generation_ref=None):
    """A zero-gradient terminal consumes its own groups without a new checkpoint."""
    rows = terminal_ranks(exit_ref, rank_refs)
    expected_step = prior_clock['successful_optimizer_steps']
    require(all(row['step'] == expected_step for row in rows), 'zero-gradient rank step differs')
    same = [{k: v for k, v in row.items() if k != 'rank'} for row in rows]
    receipt = same[0]
    require(all(row == receipt for row in same) and read(prior_checkpoint_ref)['step'] == expected_step,
            'zero-gradient outcome not unanimous on the active checkpoint')
    verify_clock(prior_clock, receipt, prior_checkpoint_ref['sha256'])
    return dict(receipt, generation_inventory_ref=generation_ref, exit_ref=exit_ref,
                rank_refs=rank_refs)


def finalized_window(cfg, hot, inventory, *, resident=False):
    """Bind the result to the exact finalized trajectory on all eight ranks."""
    from pull_closed import physical_path

    refs = [pin(hot / f'rank-{rank}' / 'FINALIZE-RESULT.json') for rank in range(8)]
    records = [read(ref) for ref in refs]
    first = records[0]
    require(all(record == first for record in records)
            and first['optimizer_steps'] == 0
            and first['source_generation_inventory_ref'] == inventory,
            'finalized generation differs across ranks or from the staged generation')
    def observed(ref):
        if resident:
            require(Path(ref['path']).resolve().is_relative_to(hot.resolve()),'result outside current resident command')
            return pin(ref)
        require(Path(ref['path']).is_relative_to('/omni-hot'), 'finalized receipt is outside the actor mount')
        path = physical_path(ref['path'], [dict(Source=str(hot), Destination='/omni-hot')])
        actual = pin(path)
        require({k:actual[k] for k in ('sha256','bytes')} == {k:ref[k] for k in ('sha256','bytes')},
                'finalized receipt differs from physical actor output')
        return actual
    trajectory_ref = observed(first['trajectory_ref'])
    require(Path(trajectory_ref['path']) == hot / cfg['generation']['runtime']['cohort_dir'] / 'COMPLETE.json',
            'finalized trajectory is outside the configured actor cohort')
    trajectory = read(trajectory_ref)
    require(trajectory['training_ready'] is True and trajectory['synthetic_fixture'] is False,
            'training consumer did not admit a real cohort')
    native_ref = observed(trajectory['native_config_ref'])
    selected_ref = observed(read(native_ref)['selected_cohort_ref'])
    selected = read(selected_ref)
    groups = selected['group_ids']
    require(selected.get('sampled_window_ref') and groups and len(groups) == len(set(groups))
            and len(groups) == len(selected['draws']), 'finalized cohort is not a sampled window')
    return trajectory_ref, selected_ref, selected, refs


def close(config_path, cfg, hot):
    """Commit the observer's original RETURN only after the actor has exited."""
    from bounded_actor_step import verified_zero_clock
    from pull_closed import physical_path

    root = hot / 'forward-return'
    if (root / 'CLOSED.json').exists():
        return read(read(root / 'CLOSED.json')['return_ref'])
    launch = read(hot / 'INTENT.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(launch['config_sha256'] == pin(config_path)['sha256'] and
            launch['boot_id'] == boot == cfg['boot_id'] and launch['devices'] == list(range(8)),
            'terminal training launch identity differs')
    exit_ref = pin(hot / 'EXIT.json')
    require(read(exit_ref)['container_id'] == read(hot / 'CONTAINER.json')['container_id'],
            'terminal receipt is for a different container')
    rank_refs = [pin(hot / f'rank-{rank}' / 'RESULT.json') for rank in range(8)]
    prior_ref = pin(Path(cfg['restore_checkpoint']['directory']) / 'COMMITTED.json')
    require(prior_ref['sha256'] == cfg['restore_checkpoint']['sha256'], 'predecessor commit drift')
    prior_clock = read(prior_ref)['clock']
    mount_cfg = cfg['capture_mounts']
    formal_roles = mount_cfg.get('formal_input_roles', [])
    mappings = [
        dict(Source=str(Path(mount_cfg['formal_input_root'] if name in formal_roles else mount_cfg['root']) / name),
             Destination=destination)
        for name, destination in mount_cfg['logical_roots'].items() if name != 'index'
    ]
    for ref in cfg['continuation']['zero_result_refs']:
        physical_ref = pin(physical_path(ref['path'], mappings))
        require(physical_ref['sha256'] == ref['sha256'] and physical_ref['bytes'] == ref['bytes'],
                'zero-gradient continuation receipt changed')
        prior_clock = verified_zero_clock(prior_clock, read(physical_ref), prior_ref['sha256'])
    inventory = cfg['generation']['generation_inventory_ref']
    trajectory_ref, selected_ref, selected, finalized_refs = finalized_window(cfg, hot, inventory)
    ranks = terminal_ranks(exit_ref, rank_refs)
    roster = []
    if ranks[0].get('status') == 'no_gradient':
        accepted = accepted_zero(exit_ref, rank_refs, prior_ref, prior_clock,
                                 verify_clock=verified_zero_clock, generation_ref=inventory)
        require(not (hot / f"step-{cfg['continuation']['prior_step'] + 1}" / 'COMMITTED.json').exists(),
                'zero-gradient receipt conflicts with an optimizer commit')
    else:
        checkpoint = hot / f"step-{cfg['continuation']['prior_step'] + 1}" / 'COMMITTED.json'
        report = hot / cfg['generation']['runtime']['cohort_dir'] / 'CONSUMER-REPORT.json'
        accepted = accepted_update(exit_ref, rank_refs, prior_ref, pin(checkpoint), pin(report),
                                   prior_clock=prior_clock, generation_ref=inventory)
        from publication import seal
        policy_ref = seal(accepted['checkpoint_ref'], root / 'publication')
        policy = read(policy_ref)
        accepted['policy_ref'] = policy_ref
        roster = [accepted['checkpoint_ref'], policy_ref, policy['publication_ref'],
                  policy['asset_manifest_ref'], *policy['assets']]
        # Recreating an inference publication validates the adapter metadata as
        # well as its two public assets. Keep every committed adapter member.
        known={r['path'] for r in roster}
        for rank in read(accepted['checkpoint_ref'])['ranks']:
            for name,expected in rank['files'].items():
                if not name.startswith('lora_adapter/'):
                    continue
                require(len(Path(name).parts)==2 and '..' not in Path(name).parts,'invalid adapter member')
                ref=pin(checkpoint.parent/name)
                require(ref['sha256']==expected,'exported adapter member changed')
                if ref['path'] not in known:
                    known.add(ref['path'])
                    roster.append(ref)
    require(accepted['group_ids'] == selected['group_ids'],
            'terminal actor did not consume the finalized group roster')
    accepted.update(trajectory_ref=trajectory_ref, selected_cohort_ref=selected_ref,
                    sampled_window_ref=selected['sampled_window_ref'])
    accepted['formal_rl_enabled'] = True
    result_ref = write(root / 'RESULT.json', accepted)
    returned = dict(optimizer_steps=accepted['optimizer_steps'], result_ref=result_ref,
                    trajectory_ref=trajectory_ref)
    for key in ('checkpoint_ref', 'policy_ref'):
        if key in accepted:
            returned[key] = accepted[key]
    return_ref = write(root / 'RETURN.json', returned)
    write(root / 'CLOSED.json', dict(schema='omni-formal-actor-return-v1', immutable=True,
          source_boot=boot, return_ref=return_ref, launch_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
          projected_members=[result_ref, exit_ref, trajectory_ref, selected_ref, *rank_refs,
                             *finalized_refs, *roster]))
    return returned


def close_live(cfg, root, prior_ref, produced_ref, rank_live_refs, peers):
    """Commit this window while the optimizer and its owner remain live."""
    from bounded_actor_step import verified_zero_clock
    from publication import seal
    root=Path(root);exchange=root.parent.parent
    intent_ref=pin(root/'COMMAND-INTENT.json');intent=read(intent_ref);owner=read(intent['owner_ref'])
    command=intent['command'];seq=command['sequence'];inventory=cfg['generation']['generation_inventory_ref']
    require(root.name==f'{seq:06d}' and owner['objects']==peers and len(peers)==len(rank_live_refs)==8
            and owner['run_id']==cfg['continuous_actor']['run_id']==command['run_id']
            and command['generation_inventory_ref']==inventory and intent['predecessor']==prior_ref
            and prior_ref['sha256']==cfg['restore_checkpoint']['sha256']==command['checkpoint_sha256']
            and command['previous_close_sha256']==pin(exchange/'READY.json' if not seq else
                root.parent/f'{seq-1:06d}'/'CLOSED.json')['sha256'],'resident command identity differs')
    refs=[];rows=[]
    for rank,live_ref in enumerate(rank_live_refs):
        folder=root/f'rank-{rank}'
        require(live_ref==pin(folder/'LIVE.json'),'resident receipt outside command/rank')
        live=read(live_ref);result_ref=pin(folder/'RESULT.json');row=read(result_ref)
        require(live==dict(rank=rank,objects=peers[rank],intent_ref=intent_ref,result_ref=result_ref)
                and row['rank']==rank,'rank or resident object identity differs')
        refs.append(result_ref);rows.append(row)
    before=read(prior_ref);clock=before['clock']
    for zero in cfg['continuation'].get('zero_result_refs',[]):
        clock=verified_zero_clock(clock,read(zero),prior_ref['sha256'])
    trajectory_ref,selected_ref,selected,finalized=finalized_window(cfg,root,inventory,resident=True)
    members=[intent_ref,trajectory_ref,selected_ref,*refs,*finalized,*rank_live_refs]
    if rows[0].get('status')=='no_gradient':
        values=[{k:v for k,v in row.items() if k!='rank'} for row in rows];receipt=values[0]
        require(produced_ref==prior_ref and all(v==receipt for v in values)
                and not (root/f"step-{before['step']+1}"/'COMMITTED.json').exists(),'invalid zero-gradient command')
        verified_zero_clock(clock,receipt,prior_ref['sha256'])
        accepted=dict(receipt,rank_refs=refs,generation_inventory_ref=inventory,resident_intent_ref=intent_ref)
    else:
        require(produced_ref==pin(root/f"step-{before['step']+1}"/'COMMITTED.json'),'wrong local recovery checkpoint')
        accepted=_updated(rows,refs,prior_ref,produced_ref,pin(root/cfg['generation']['runtime']['cohort_dir']/
            'CONSUMER-REPORT.json'),prior_clock=clock,generation_ref=inventory,terminal=dict(resident_intent_ref=intent_ref))
        accepted['policy_ref']=seal(produced_ref,root/'publication',fresh_commit=True)
        policy=read(accepted['policy_ref'])
        members.extend([produced_ref,accepted['policy_ref'],policy['publication_ref'],policy['asset_manifest_ref']])
        for entry in read(produced_ref)['ranks']:
            for name,sha in entry['files'].items():
                if name.startswith('lora_adapter/'):
                    require(len(Path(name).parts)==2 and '..' not in Path(name).parts,'invalid adapter path')
                    asset=pin(Path(produced_ref['path']).parent/name)
                    require(asset['sha256']==sha,'adapter differs from committed export')
                    members.append(asset)
    require(accepted['group_ids']==selected['group_ids'],'resident result differs from sampled roster')
    result=write(root/'RESULT.json',dict(accepted,formal_rl_enabled=True,selected_cohort_ref=selected_ref,sampled_window_ref=selected['sampled_window_ref'],
                                         trajectory_ref=trajectory_ref))
    returned=dict(optimizer_steps=accepted['optimizer_steps'],result_ref=result,trajectory_ref=trajectory_ref,
                  **{k:accepted[k] for k in ('checkpoint_ref','policy_ref') if k in accepted})
    return write(root/'CLOSED.json',dict(schema='formal-resident-command-v1',immutable=True,source_boot=owner['boot_id'],
        sequence=seq,run_id=owner['run_id'],rank_live_refs=rank_live_refs,result_ref=result,projected_members=[result,*members],
        return_ref=write(root/'RETURN.json',returned)))
