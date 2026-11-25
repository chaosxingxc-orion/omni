"""Full screen selection -> complete selected fresh release, using original sampler."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
from cohort_io import BASE,ACL,CODE,HERE,read,raw,save,pin,require,digest,imports,load,storage,environment,verify_parents


# Physical group storage only. The consumer sees the original per-call payloads.
# Do not put policy-dependent fields, full-sequence positions or labels here.
STATIC_FIELDS = ('audio_features', 'feature_mask', 'effective_audio_lengths',
                 'prompt_token_ids', 'prompt_messages', 'prompt_text')


def group_storage(group, *, expand=False):
    """Intern exact, canonical field values within this draw; preserve distinct prompts."""
    if expand and 'retained_static_v1' not in group:
        return group
    table = group.pop('retained_static_v1') if expand else {}
    used = set()
    require(isinstance(table, dict), 'invalid group static table')
    if expand:
        require(all(digest(value) == key for key, value in table.items()), 'changed group static field')
    # Assemble creates one audio object per conditioning identity. Hash that
    # object once, retaining a strong reference for the duration of the call.
    memo = {}
    for g in group['online']['behavior_bundle']['groups']:
        for episode in g['episodes']:
            for call in episode['calls']:
                f = call['training_fields']
                if expand:
                    keys = f.pop('retained_static_refs')
                    require(set(keys) == set(STATIC_FIELDS) and not set(keys).intersection(f),
                            'invalid group static bindings')
                    for name, key in keys.items():
                        f[name] = table[key]
                        used.add(key)
                else:
                    require('retained_static_refs' not in f, 'already interned')
                    keys = {}
                    for name in STATIC_FIELDS:
                        value = f.pop(name)
                        if id(value) not in memo:
                            key = digest(value)
                            memo[id(value)] = value, key
                            table.setdefault(key, value)
                        keys[name] = memo[id(value)][1]
                    f['retained_static_refs'] = keys
    if expand:
        require(used == set(table), 'unused group static fields')
    else:
        group['retained_static_v1'] = table
    return group


def save_training_rows(path, rows, encoding):
    """Durable immutable JSON. Compact encoding changes only whitespace and its file pin."""
    if encoding == 'original':
        return save(path, rows)
    require(encoding == 'compact_json_v1', 'unknown training row encoding')
    import hashlib, os, tempfile
    storage(path.parent)
    # Keep memory bounded to one recording, and hash exactly the bytes we publish.
    digestor=hashlib.sha256()
    size=0
    with tempfile.NamedTemporaryFile(dir=path.parent) as out:
        def put(raw):
            nonlocal size
            out.write(raw); digestor.update(raw); size+=len(raw)
        put(b'[')
        for i,row in enumerate(rows):
            if i:put(b',')
            put(json.dumps(row,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
        put(b']\n')
        out.flush();os.fsync(out.fileno())
        expected=dict(path=str(path.resolve()),bytes=size,sha256=digestor.hexdigest())
        try:os.link(out.name,path)
        except FileExistsError:
            require(pin(path)==expected,'immutable training rows changed')
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return expected


def draw(config,root,*,fixture=False,persist=True):
    sampling=copy.deepcopy(config['sampling'])
    from policy import validate_scope
    validate_scope(config,fixture=fixture)
    manifest=read(config['screen_ref'])
    panel=manifest.get('schema')=='node-panel-selection-v1'
    if panel:
        from node_screen_selection import panel_binding
        panel_binding(manifest,sampling)
        require(config.get('sampled_window_ref'),'panel must consume the sealed selection')
    else:
        from node_screen_selection import outcome_mix
        require(sampling['purpose']=='screen','screen sampling required')
        outcome_mix(sampling)
    require(manifest['version']==sampling['pool_version'],'screen version mismatch')
    if config.get('sampled_window_ref'):
        window=read(config['sampled_window_ref']);dataset=read(window['dataset_ref'])
        selected=read(dataset['selected_ref'])
        require(window['schema'] in ('frozen-query-window-v1','reference-query-window-v1') and dataset['schema']=='frozen-sampled-query-dataset-v1','frozen sampled window schema')
        require(window['screen_ref']==dataset['screen_ref']==config['screen_ref'],'frozen window screen mismatch')
        require(sampling==dataset['sampling'],'frozen selection sampling identity differs')
        start,end=window['start'],window['end']
        require(type(start) is int and type(end) is int and 0<=start<end<=dataset['count']==len(selected['draws']),'frozen window bounds')
        from training_references import window_draws
        draws=window_draws(window,dataset,selected)
        for index,item in enumerate(draws):
            item['dataset_draw_index']=start+index
            item['draw_index']=index
            item.pop('query_draw_id',None);item.pop('group_id',None)
        stats=dict(samples=len(draws),frozen_dataset_statistics=dataset['statistics'],
                   start=start,end=end,selection_probability_scope='full_frozen_dataset')
    else:
        draws,stats=sample_original(manifest,sampling)
    locked_node=(manifest.get('schema')=='node-screen-selection-v1' and config.get('sampled_window_ref') and
                 manifest.get('node')==sampling.get('node')=='P' and sampling.get('scope')=='node' and
                 manifest.get('retrieval_population_complete') is True and manifest.get('sealed') is True and
                 manifest.get('official_train_count',0)>0 and manifest.get('synthetic_fixture') is False and
                 selected['statistics'].get('inputs_ref')==manifest['inputs_ref'] and
                 manifest['version']==manifest['inputs_ref']['sha256'])
    if panel:
        require(selected['statistics']['inputs_ref']==manifest['inputs_ref'] and
                selected['statistics']['population_scope']==manifest['population_scope'],'window not from sealed panel')
    require(fixture or panel or locked_node or (manifest['official_train_count']>0 and not manifest['synthetic_fixture'] and manifest['population_complete'] and manifest.get('unknown_count',0)==0),'complete real TRAIN screen required')
    require(fixture or all(not d['record'].get('synthetic_fixture') for d in draws),'fixture cannot enter real capture')
    identity=dict(screen=config['screen_ref'],sampling=sampling,policy=config['policy'],clock=config['clock_state_ref'],cohort_id=config.get('cohort_id',config['native']['attempt_id']),capture=config['capture'],learning=config['learning_config_ref'],consumer=config['consumer_config_ref'],native=config['native'],implementation={p.name:pin(p) for p in sorted(HERE.glob('*.py'))})
    if config.get('sampled_window_ref'):identity['sampled_window']=config['sampled_window_ref']
    version=digest(identity)
    for d in draws:
        d['query_draw_id']=version+':draw:'+str(d['draw_index'])
        d['group_id']=version+':group:'+str(d['draw_index'])
    plan=dict(schema='selected-fresh-cohort-v1',version=version,screen_ref=config['screen_ref'],
        full_screen_population=manifest['population'] if panel else manifest['official_train_count'],sampling=sampling,draws=draws,
        group_ids=[d['group_id'] for d in draws],statistics=stats,policy=config['policy'],
        training_ready=False,synthetic_fixture=fixture,
        completion_contract='all selected draw groups; no new outcome filtering, ineligible rewriting, replacement or refill')
    if panel:plan['screen_population_scope']=manifest['population_scope']
    if config.get('sampled_window_ref'):plan['sampled_window_ref']=config['sampled_window_ref']
    if persist:save(Path(root)/'SELECTED.json',plan)
    return plan


def sample_original(manifest,sampling):
    with imports([CODE,BASE/'collection-dispatch-integration'],('common','kernel','stream_pool','base','sampling','consumer')):
        import stream_pool
        sampler=stream_pool.sampler_module(BASE/'unified-learning')
        if manifest.get('compatible_runtime_systems'):
            original=load(BASE/'collection-dispatch-integration/sampling.py','fresh_screen_sampling')
            import types
            function=types.FunctionType(sampler.sample.__code__,dict(sampler.__dict__,check_pool=original.check_compatible_pool))
        else:function=sampler.sample
        pool=stream_pool.Pool(manifest) if 'streams' in manifest else manifest
        difficulty=sampling.get('node_difficulty')
        if difficulty is not None:
            require(sampling['purpose']=='screen','node difficulty is historical input selection only')
            from node_difficulty import projection,restore
            import types
            original_check=function.__globals__['check_pool']
            records,pool_excluded,metadata=original_check(pool,sampling)
            projected,originals,excluded=projection(records,manifest,difficulty)
            # Run the existing stratified algorithm without rescanning the pool.
            # Preserve official population/outcome accounting from its validator.
            def checked_view(unused_pool,unused_config):
                return projected,pool_excluded,metadata
            function=types.FunctionType(function.__code__,dict(function.__globals__,check_pool=checked_view))
            draws,stats=function(pool,sampling,'.')
            return restore(draws,stats,originals,difficulty,excluded)
        draws,stats=function(pool,sampling,'.')
    return draws,stats


def clock_for(config):
    from cohort_io import imports
    with imports([ACL,BASE/'native-training-successor']):
        from native_update import configure_imports
        configure_imports()
        from task_runtime import task_module
        return task_module().create_clock(read(config['learning_config_ref']),read(config['clock_state_ref']),consumer_config=read(config['consumer_config_ref']))


def prepare(config,root,*,fixture=False):
    verify_parents();root=Path(root);root.mkdir(parents=True,exist_ok=True);storage(root)
    require(config['schema']=='configured-fresh-cohort-v1' and config['progress_check_seconds']==900,'single configured cohort/cadence required')
    plan=draw(config,root,fixture=fixture)
    clock=clock_for(config)
    if not fixture:
        from policy import validate
        validate(config['policy'],read(config['learning_config_ref']),clock.state)
    require(config.get('rollout_scope',plan['sampling']['scope'])==clock.phase()['rollout_scope'],'scope controlled by sole learning Clock')
    for gid in plan['group_ids']:
        require(gid not in clock.state['groups'] or not clock.state['groups'][gid]['consumed'],'selected group already consumed')
        clock.reserve(gid)
    save(root/'RESERVED-CLOCK.json',clock.state)
    cfg=read(config['consumer_config_ref'])
    cfg.update(learning_config_ref=config['learning_config_ref'],clock_state_ref=pin(root/'RESERVED-CLOCK.json'),selected_cohort_ref=pin(root/'SELECTED.json'))
    save(root/'CONSUMER.json',cfg)
    save(root/'INPUT.json',config)
    if not fixture:
        pending=save(root/'PENDING-CAPTURE.json',dict(input_ref=pin(root/'INPUT.json'),cohort_root=str(root),selected_cohort_ref=pin(root/'SELECTED.json')))
        native=copy.deepcopy(config['native'])
        native.update(schema='native-configured-training-v1',enabled=False,pending_capture_ref=pending,consumer_source_ref=pin(HERE/'consumer.py'),learning_config_ref=config['learning_config_ref'],clock_state_ref=pin(root/'RESERVED-CLOCK.json'),consumer_config_ref=pin(root/'CONSUMER.json'),selected_cohort_ref=pin(root/'SELECTED.json'))
        bind_migration(native,root)
        save(root/'NATIVE.pending.disabled.json',native)
    return plan,clock,cfg


def publish(config,root,*,fixture=False,consumer_ref=None,accepted_sink=None,row_encoding="original"):
    """All selected groups, exact order; consumer is authoritative for full G."""
    consumer=load(HERE/'consumer.py','fresh_release_consumer')
    build,restore=consumer.build,consumer.restore
    root=Path(root);plan=read(pin(root/'SELECTED.json'));consumer_ref=consumer_ref or pin(root/'CONSUMER.json');cfg=read(consumer_ref)
    indexed=config['capture'].get('training_input_layout')=='group_refs_v1'
    if indexed:require(accepted_sink is not None and not fixture,'indexed groups require accepted live actor')
    group_refs=[]
    online=[];offline=[];provenance=[];costs=[];task_records=[]
    for d in plan['draws']:
        path=root/'groups'/str(d['draw_index'])/'GROUP.json'
        require(path.is_file(),'incomplete selected cohort: keep partial groups and reconcile; no refill')
        group_ref=pin(path)
        if indexed:group_refs.append(group_ref)
        captured=group_storage(read(group_ref), expand=True)
        require(captured['group_id']==d['group_id'] and captured['query_draw_id']==d['query_draw_id'],'capture/draw binding')
        require(captured['policy']==plan['policy'],'capture policy versions changed')
        require(captured['synthetic'] is fixture,'fixture provenance mismatch')
        row=captured['online'];row.update(draw_index=d['draw_index'],sample_id=d['record']['sample_id'])
        off=captured['offline'];off.update(draw_index=d['draw_index'],sample_id=d['record']['sample_id'])
        online.append(row);offline.append(off)
        p={k:v for k,v in d.items() if k!='record'}
        p.update({k:d['record'][k] for k in ('sample_id','recording_ref_id','source_id','source','domain','outcome')})
        provenance.append(p);costs.append(captured['costs']);task_records.extend(captured.get('task_records',[]))
    rows=dict(online=online,offline=offline,provenance=provenance)
    refs={n:save_training_rows(root/(n+'.json'),v,row_encoding) for n,v in rows.items() if not indexed or n!='online'}
    if indexed:
        refs['online']=save(root/'online.json',dict(schema='group-online-index-v1',
            selected_ref=cfg['selected_cohort_ref'],groups=group_refs))
    files=dict(count=len(online),refs=refs,**{n:r['path'] for n,r in refs.items()})
    if accepted_sink is not None:
        if indexed:files['layout']='group_refs_v1'
        fields,report,bundle=consumer.build_with_bundle(files,cfg,**({'publisher_groups':(group_refs,online)} if indexed else {}))
    else:
        fields,report=build(files,cfg)
        bundle,_=restore(files,cfg)
    require(set(g['group_id'] for g in bundle['groups'])==set(plan['group_ids']),'complete selected group accounting')
    save(root/'FILES.json',files);save(root/'CAPTURE-COSTS.json',costs)
    save(root/'CONSUMER-REPORT.json',report);save(root/'TASK-RECORDS.private.json',task_records)
    ready=not fixture and not bundle['synthetic'] and report['required_fields_validated']
    native=copy.deepcopy(config['native'])
    native.update(schema='native-configured-training-v1',enabled=False,
        consumer_source_ref=pin(HERE/'consumer.py'),fresh_group_release_ref=pin(root/'FILES.json'),
        consumer_config_ref=consumer_ref,learning_config_ref=config['learning_config_ref'],
        clock_state_ref=pin(root/'RESERVED-CLOCK.json'),selected_cohort_ref=pin(root/'SELECTED.json'),
        capture_costs_ref=pin(root/'CAPTURE-COSTS.json'))
    bind_migration(native,root)
    # Existing native task compiler owns observed labels; no hand-authored spans.
    task_inputs(config,root,bundle,report,native)
    save(root/'NATIVE.disabled.json',native)
    result=dict(schema='selected-fresh-dataset-v1',version=digest(dict(plan=pin(root/'SELECTED.json'),files=refs,clock=pin(root/'RESERVED-CLOCK.json'))),
        training_ready=ready,native_execution_enabled=False,synthetic_fixture=fixture,
        full_screen_population=plan['full_screen_population'],selected_groups=len(online),
        complete_episodes=sum(len(g['episodes']) for g in bundle['groups']),
        gradient_status=report['gradient_status'],global_episode_denominator=report['global_episode_denominator'],
        files_ref=pin(root/'FILES.json'),consumer_config_ref=consumer_ref,
        native_config_ref=pin(root/'NATIVE.disabled.json'),capture_costs_ref=pin(root/'CAPTURE-COSTS.json'))
    if 'screen_population_scope' in plan:result['screen_population_scope']=plan['screen_population_scope']
    save(root/'COMPLETE.json',result)
    if accepted_sink is not None:
        require(ready,'only accepted real fields may be reused')
        accepted_sink(root,fields,report)
    return result


def bind_migration(native,root):
    if native.get('policy_config_migration_ref'):
        migration=read(native['policy_config_migration_ref'])
        for key in ('learning_config_ref','clock_state_ref','consumer_config_ref'):migration[key]=native[key]
        migration['selected_cohort_ref']=native['selected_cohort_ref']
        native['policy_config_migration_ref']=save(Path(root)/'SELECTED-POLICY-MIGRATION.json',migration)


def task_inputs(config,root,bundle,report,native):
    """Compile original task sidecar from retained annotations when available."""
    import contextlib
    stack=contextlib.ExitStack()
    official_binding=None
    rows=[];online=[]
    retained={tuple(r[k] for k in ('group_id','episode_id','call_id','node')):r for r in read(pin(root/'TASK-RECORDS.private.json'))}
    actor_keys={(m['group_id'],m['episode_id'],m['call_id'],m['node']) for m in report['all_call_mapping'] if m['actor_row'] is not None}
    try:
        for g in bundle['groups']:
            for e in g['episodes']:
                for c in e['calls']:
                    ident=dict(group_id=g['group_id'],episode_id=e['episode_id'],call_id=c['call_id'],node=c['node'])
                    if tuple(ident.values()) not in actor_keys:continue
                    fact=retained.get(tuple(ident.values()))
                    if fact:
                        from official import sources
                        if official_binding is None:official_binding=stack.enter_context(sources(config))
                        official_sources,official_adapter=official_binding
                        row,source=official_adapter.from_official(fact['annotation'],fact['packet'],sources=official_sources,reference_id=fact['annotation']['recording_ref_id'])
                        row.update(ident)
                    else:
                        require(all(v==0 for v in report['snapshot'].get('auxiliary_coefficients',{}).values()),'observed auxiliary supervision requires actual official annotation/packet')
                        row=dict(ident,state='missing');source=dict(packet=c.get('actual_packet',dict(stage=c['node'])))
                    rows.append(row);online.append(dict(identity=ident,online_input=source))
    finally:stack.close()
    sidecar=dict(schema='task-label-sidecar-v1',boundary='offline_only',split='TRAIN',source_ref=pin(root/'offline.json'),
        system_version=read(config['learning_config_ref'])['system_version'],snapshot_sha256=report['snapshot']['snapshot_sha256'],records=rows)
    native.update(task_sidecar_ref=save(root/'TASK-SIDECAR.json',sidecar),online_by_call_ref=save(root/'ONLINE-BY-CALL.json',online))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--operation',choices=['prepare','capture','publish','run','native-plan'],default='prepare');p.add_argument('--enable-model-requests',action='store_true');p.add_argument('--enable-native-training',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_bytes());root=Path(config['output_root']).resolve();environment(root)
    with (root/'COHORT.guard').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.operation in ('prepare','run'):prepare(config,root)
        if a.operation in ('capture','run'):
            if not a.enable_model_requests:
                print(json.dumps(dict(status='PREPARED_REQUESTS_DISABLED',training_ready=False)));return
            from cohort_capture import capture
            from execution import CaptureLease,arm
            with CaptureLease(config,root) as lease:capture(config,root,execution_enabled=True,lease=lease)
            if a.enable_native_training:
                from launch_native import launch
                raise SystemExit(launch(config))
            print(json.dumps(arm(config,root,enabled=False)))
        elif a.operation=='native-plan':
            require(not a.enable_native_training,'native-plan is disabled; use launch_native for real guarded execution')
            from execution import arm
            print(json.dumps(arm(config,root,enabled=False)))
        elif a.operation=='publish':print(json.dumps(publish(config,root)))
if __name__=='__main__':main()
