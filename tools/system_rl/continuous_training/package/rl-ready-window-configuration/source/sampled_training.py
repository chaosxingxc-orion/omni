"""Frozen input membership and exact consumption for the original RL controller.

These are historical query draws, never cached policy actions. Each window still
uses the original fresh G capture, Clock, optimizer and durable request ledger.
"""
import copy
from pathlib import Path
from dependencies import require
from run_package import read, pin, write


def specification(config, *, prior_step=0):
    spec = config.get('sampled_training')
    if spec is None:
        return None
    require(spec['schema'] == 'frozen-sampled-training-v1', 'sampled TRAIN schema')
    require(config['execution_mode'] == 'formal' and not config.get('bounded_restore_compare'),
            'sampled TRAIN completion belongs to the formal workflow')
    require(type(spec['sample_count']) is int and spec['sample_count'] > 0
            and type(spec['queries_per_cohort']) is int and spec['queries_per_cohort'] > 0,
            'positive fixed sample count and cohort size')
    require(spec['resample'] == 'after_current_dataset_complete'
            and spec['zero_signal'] == 'consume_without_optimizer_step', 'explicit completion semantics')
    # Existing storage/worker ceiling remains a safety envelope. It must cover
    # the worst case (one successful update for each selected window).
    maximum = (spec['sample_count'] + spec['queries_per_cohort'] - 1) // spec['queries_per_cohort']
    require(type(prior_step) is int and prior_step >= 0, 'invalid initial optimizer step')
    require(config['max_optimizer_updates'] >= prior_step + maximum,
            'update storage envelope cannot truncate sampled TRAIN after the initial checkpoint')
    return spec


def prepare(config, state, root):
    spec = specification(config)
    if spec is None:
        return
    if state.get('sampled_dataset_ref'):
        dataset = read(state['sampled_dataset_ref'])
        require(dataset['specification'] == spec, 'sampled TRAIN specification drift')
        require(dataset['screen_ref'] == state['screen']['screen_ref'], 'screen changed during sampled TRAIN')
        return
    marker = Path(root) / 'sampled-training/DATASET.json'
    if marker.exists():
        dataset = read(marker)
        require(dataset['specification'] == spec and dataset['screen_ref'] == state['screen']['screen_ref'],
                'partial sampled TRAIN identity drift')
    else:
        source = copy.deepcopy(read(state['input_ref']))
        pool = read(state['screen']['screen_ref'])
        source['screen_ref'] = state['screen']['screen_ref']
        source['sampling'].update(pool_version=pool['version'], system_version=state['screen']['system_version'],
                                  sample_count=spec['sample_count'])
        scoped=pool.get('schema') in ('node-screen-selection-v1','node-panel-selection-v1')
        if not scoped:source['sampling']['seed']=config['root_seed']
        source.pop('sampled_window_ref', None)
        # Only the existing sampler runs here, no requests, clock reservations,
        # tensors, optimizer or synthetic actions are constructed.
        from fresh_adapter import scope
        frozen=None
        if scoped:
            from node_screen_selection import selection as node_selection
            draws,statistics=node_selection(pool,source['sampling'])
            learning=read(source['learning_config_ref']);clock=read(source['clock_state_ref'])
            phase=[p for p in learning['phases'] if p['start_step']<=clock['successful_optimizer_steps']][-1]
            from node_screen_selection import phase_inputs
            frozen=phase_inputs(source,phase,pool)
        else:
            with scope() as cohort:
                from node_difficulty import prepare_selection
                source['sampling']=prepare_selection(source['sampling'],source['screen_ref'],
                                                      marker.parent/'node-difficulty')
                # Freeze query membership only. draw() also rewrites node-specific
                # upstream packets and SELECTED.json; invoking it here made a
                # restart between SELECTED and DATASET collide with that rewrite.
                draws, statistics = cohort.sample_original(pool, source['sampling'])
                if source.get('phase_frozen_upstream_ref'):
                    learning=read(source['learning_config_ref']);clock=read(source['clock_state_ref'])
                    phase=[p for p in learning['phases'] if p['start_step']<=clock['successful_optimizer_steps']][-1]
                    if phase['rollout_scope']=='node':
                        frozen=pin(source['phase_frozen_upstream_ref'])
                        require(frozen['sha256']==phase['frozen_upstream_release'],'node packet release drift')
                        from cohort_io import imports,BASE
                        from training_references import bind_frozen_inputs
                        with imports([BASE/'collection-scale-successor/code'],('common','stream_pool')):
                            from stream_pool import Pool
                            upstream=read(frozen)
                            draws=bind_frozen_inputs(draws,Pool(upstream)['records'] if 'streams' in upstream else upstream['records'])
                if source.get('prepare_mode')=='full_train_screen':
                    from material import expand
                    from recording_inputs import bind_origins
                    draws=bind_origins(draws,marker.parent/'recordings',expand)
        require(len(draws) == spec['sample_count'], 'incomplete original sampling')
        from training_references import compact
        selected = compact(draws, marker.parent / 'references')
        selected['statistics'] = statistics
        selection = write(marker.parent / 'selection/SELECTED.json', selected)
        dataset = dict(schema='frozen-sampled-query-dataset-v1', specification=spec,
                       screen_ref=source['screen_ref'], input_ref=state['input_ref'],
                       selected_ref=selection,
                       count=len(draws), sampling=source['sampling'], statistics=statistics,
                       fresh_actions=False, training_ready=False)
        if frozen:dataset['phase_frozen_upstream_ref']=frozen
        if source.get('prepare_mode')=='full_train_screen':dataset['selected_recording_origins']=True
        write(marker, dataset)
    state['sampled_dataset_ref'] = pin(marker)
    state['sampled_consumed'] = 0
    state['sampled_windows'] = []


def complete(config, state):
    if specification(config) is None or not state.get('sampled_dataset_ref'):
        return False
    return state['sampled_consumed'] == read(state['sampled_dataset_ref'])['count']


def bind_phase_cache(state, source, output):
    """Refresh upstream contexts once, without redrawing or resetting membership.

    The immutable dataset successor keeps sampling probabilities and the consumed
    cursor. Only online inputs change at an explicitly bound node phase boundary.
    This CPU preparation runs before capture/resource admission.
    """
    dataset = read(state['sampled_dataset_ref'])
    release = source.get('phase_frozen_upstream_ref')
    if not release:
        return dataset
    learning = read(source['learning_config_ref']); clock = read(source['clock_state_ref'])
    phase = [p for p in learning['phases'] if p['start_step'] <= clock['successful_optimizer_steps']][-1]
    if phase['rollout_scope'] != 'node':
        return dataset
    release = pin(release)
    require(release['sha256'] == phase['frozen_upstream_release'], 'phase cache release drift')
    if dataset.get('phase_frozen_upstream_ref') == release:
        return dataset
    root = Path(output) / 'phase-context' / release['sha256']
    marker = root / 'DATASET.json'
    prior = pin(state['sampled_dataset_ref'])
    if marker.exists():
        updated = read(marker)
        require(updated['previous_context_dataset_ref'] == prior and
                updated['phase_frozen_upstream_ref'] == release, 'phase cache restart identity drift')
    else:
        from training_references import expand, compact, bind_frozen_inputs
        selection = read(dataset['selected_ref'])
        draws = expand(selection['draws'])
        from fresh_adapter import scope
        with scope():
            from cohort_io import imports, BASE
            with imports([BASE/'collection-scale-successor/code'], ('common', 'stream_pool')):
                from stream_pool import Pool
                pool = read(release)
                draws = bind_frozen_inputs(draws, Pool(pool)['records'] if 'streams' in pool else pool['records'])
        selected = compact(draws, root/'references')
        selected['statistics'] = dataset['statistics']
        updated = dict(dataset, selected_ref=write(root/'SELECTED.json', selected),
                       phase_frozen_upstream_ref=release, previous_context_dataset_ref=prior,
                       membership_dataset_ref=dataset.get('membership_dataset_ref', prior))
        write(marker, updated)
    state['sampled_dataset_ref'] = pin(marker)
    return updated


def window_capacity(config, source, count):
    """Compute the real request ceiling for this recording window (also for the tail)."""
    if not config.get('formal_actor'):
        return
    from prompt_capacity import sampled_envelope
    sampling=source['sampling'];scope=source['rollout_scope']
    require(sampling['scope']==scope and 0<count<=config['sampled_training']['queries_per_cohort'],
            'sampling scope or window size differs')
    b=source['budget']
    calls,tokens,caps=sampled_envelope(config['formal_rollout']['window_capacity'],b,count,
        read(source['learning_config_ref'])['kernel']['G'],scope,sampling.get('node'))
    stage=config['stage_budgets']['capture']
    require(stage['requests_upper']>=calls and stage['tokens_upper']>=3*tokens,
            'capture reservation does not cover sampled requests, current and reference')
    b.update(max_requests=calls,capture_logical_tokens_upper=tokens)
    b.pop('sampled_request_limit',None);b.pop('sampled_request_limits',None)
    if scope=='node':
        b['sampled_request_limit']=dict(node=sampling['node'],max_tokens=caps[sampling['node']])
    else:
        b['sampled_request_limits']=caps


def bind_window(config, state, input_ref, output):
    if specification(config) is None:
        return input_ref
    source = copy.deepcopy(read(input_ref))
    dataset = bind_phase_cache(state, source, output)
    selected = read(dataset['selected_ref'])
    start = state['sampled_consumed']
    end = min(start + dataset['specification']['queries_per_cohort'], dataset['count'])
    require(0 <= start < end <= len(selected['draws']), 'sampled TRAIN cursor outside remaining inputs')
    # Probability/multiplicity belongs to the FULL frozen selection, not to
    # separately rounded per-window 80/20 quotas.
    source['sampling'] = copy.deepcopy(dataset['sampling'])
    window_capacity(config, source, end-start)
    compact = selected.get('schema') == 'reference-query-selection-v1'
    window = dict(schema='reference-query-window-v1' if compact else 'frozen-query-window-v1', dataset_ref=state['sampled_dataset_ref'],
                  screen_ref=dataset['screen_ref'], start=start, end=end,
                  **({} if compact else dict(draws=copy.deepcopy(selected['draws'][start:end]))))
    ref = write(Path(output) / 'WINDOW.json', window)
    source['sampled_window_ref'] = ref
    source['policy']['formal_sampled'] = True
    if dataset.get('selected_recording_origins'):
        from recording_inputs import window_mapping
        mapping=window_mapping(selected['draws'][start:end],Path(output)/'recordings',dataset_ref=state['sampled_dataset_ref'])
        consumer=read(source['consumer_config_ref']);consumer['recording_map_ref']=mapping
        source['consumer_config_ref']=write(Path(output)/'CONSUMER.json',consumer)
    state['sampled_pending_window_ref'] = ref
    return write(Path(output) / 'INPUT.json', source)


def consume(config, state, result):
    if specification(config) is None:
        return
    ref = state['sampled_pending_window_ref']
    window = read(ref)
    require(window['dataset_ref'] == state['sampled_dataset_ref']
            and window['start'] == state['sampled_consumed'], 'out-of-order or duplicate sampled consumption')
    require(type(result['optimizer_steps']) is int and result['optimizer_steps'] in (0, 1)
            and result.get('result_ref'), 'only an actual terminal training result may consume inputs')
    terminal = read(result['result_ref'])
    require(terminal.get('clock') is not None, 'terminal original Clock missing')
    if 'selected_cohort_ref' in terminal:
        require(terminal['formal_rl_enabled'] is True
                and terminal['trajectory_ref'] == result['trajectory_ref'] == state['trajectory_ref']
                and terminal['sampled_window_ref'] == ref, 'formal terminal trajectory differs')
        selected = read(terminal['selected_cohort_ref'])
        require(terminal['group_ids'] == selected['group_ids']
                and terminal['clock']['attempts'][terminal['attempt_id']] ==
                dict(group_ids=selected['group_ids'], succeeded=result['optimizer_steps']==1),
                'formal Clock did not commit this finalized cohort')
    else:
        native = read(read(state['trajectory_ref'])['native_config_ref'])
        selected = read(native['selected_cohort_ref'])
    require(selected['sampled_window_ref'] == ref and len(selected['draws']) == window['end']-window['start'],
            'training result consumed a different selected window')
    require(all(gid in terminal['clock']['groups'] and terminal['clock']['groups'][gid]['consumed']
                for gid in selected['group_ids']), 'all fresh groups must reach terminal consumption')
    state['sampled_windows'].append(dict(window_ref=ref, result_ref=result['result_ref'],
                                        optimizer_steps=result['optimizer_steps']))
    state['sampled_consumed'] = window['end']
    state.pop('sampled_pending_window_ref')
