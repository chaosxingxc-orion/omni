"""Closed real assets + original program + live authority -> original finite input.

No sampler, endpoint, model loading, or resource lease is implemented here.
The owner endpoint stage supplies the actual witness; DATA supplies real sources.
"""
import copy,datetime
from pathlib import Path
from dependencies import ROOT,HERE,require,upstream
from run_package import read,pin,write
from policy_identity import digest


def bind_business_executables(business, root):
    """Derive active code pins from the sealed release; never repin data/history."""
    release_ref=pin(HERE/'SOURCE-PINS.json')
    sources=read(release_ref)['sources']
    bypath={row['path']:row for row in sources}
    changes=[]
    for index,previous in enumerate(business.get('pins',[])):
        path=Path(previous['path'])
        if path.suffix!='.py':
            continue
        from dependencies import VIRTUAL_ROOT
        if not path.is_relative_to(VIRTUAL_ROOT):
            pin(previous)  # External frozen package retains its original identity.
            continue
        declared=bypath.get(str(path))
        require(declared is not None,'business executable absent from sealed release: '+str(path))
        current=pin(declared)
        require(current==declared,'sealed business executable identity differs')
        if previous!=current:
            business['pins'][index]=current
            changes.append(dict(previous=previous,current=current))
    write(Path(root)/'EXECUTABLE-MIGRATION.json',dict(
        schema='active-business-executable-migration-v1',release_ref=release_ref,
        changes=changes,data_and_knowledge_pins_unchanged=True))
    return business


def bind_preparation_authority(business_ref, root):
    """Version mutable policy-document bindings without changing scientific inputs.

    Mirrors the existing evaluation runner's authority migration. Snapshots are
    immutable inputs, not a grant of model/host execution authority.
    """
    business=copy.deepcopy(read(business_ref))
    bind_business_executables(business,root)
    bind_materializer_authority(business,root)
    return write(Path(root)/'BUSINESS.json',business)


def bind_materializer_authority(business,root):
    """Shared authority-only derivation; no data or knowledge repinning."""
    import hashlib
    from dependencies import local_path
    original=business['encoder']['p_materializer_config']
    materializer=copy.deepcopy(read(original))
    prior=materializer['inputs']['authority']
    current=[];evidence=[]
    root=Path(root)
    for item in prior:
        actual=pin(item['path'])
        raw=local_path(actual['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==actual['sha256'],'authority changed while snapshotting')
        # JSON envelope is stable under the original durable writer. Materializer
        # authenticates authority bytes; it does not interpret their contents.
        # The envelope contains source_path: equal contents at different source
        # paths are distinct documents, and must not collide in the store.
        snapshot_key=digest(dict(path=actual['path'],sha256=actual['sha256']))
        snapshot=write(root/'authority'/(snapshot_key+'.json'),dict(
            schema='immutable-authority-document-snapshot-v1',
            source_path=actual['path'],source_sha256=actual['sha256'],text=raw.decode()))
        current.append(snapshot)
        evidence.append(dict(previous=item,current=actual,snapshot=snapshot))
    materializer['inputs']['authority']=current
    materializer['readonly_roots']=list(dict.fromkeys(materializer['readonly_roots']+[str(root.resolve())]))
    pmref=write(root/'P-MATERIALIZER.json',materializer)
    business['encoder']['p_materializer_config']=pmref
    write(root/'AUTHORITY-MIGRATION.json',dict(original_materializer_ref=original,
        documents=evidence,scientific_inputs_unchanged=True,execution_authorization=False))
    return business


def bind_official_source(source, root, template_ref=None):
    """Bind original TRAIN authority and a stage-local persistent index cache."""
    ref = source['capture'].get('official_source_config_ref') or template_ref
    if ref is None:
        ref = pin(ROOT/'task-loss-integration/official-source-successor/config.official-train.json')
    official = copy.deepcopy(read(ref))
    require(official['schema']=='official-train-task-source-v1' and official['split']=='TRAIN',
            'official TRAIN source required')
    require({d['domain'] for d in official['domains']}=={'train_real','train_synth'},
            'both official TRAIN domains required')
    official['index_cache_root'] = str(Path(root).resolve()/'official-source-index')
    source['capture']['official_source_config_ref'] = write(Path(root)/'OFFICIAL-SOURCE.json', official)
    return source


def build(config,output):
    root=Path(output);spec=config['runtime_construction'];marker=root/'BUILT.json'
    mode=config.get('execution_mode','finite_engineering')
    identity=dict(execution_mode=mode,construction=spec,finite_config=config['finite_capture_binding_ref'],registry=config['registry'],
        authority=config['authorization_ref'],program=config['program_template_ref'],deadline=config['deadline_epoch'],
        budget=config['budget'],run_id=config['run_id'])
    if marker.exists():
        result=read(marker);require(result['identity']==identity,'runtime construction identity drift')
        pin(result['input_ref']);return result
    require(spec['schema']=='finite-real-runtime-construction-v1','real runtime construction required')
    seal_ref=pin(spec.get('base_asset_manifest_ref',ROOT/'policy-iteration-successor/BASE-ASSETS.json'))
    seal=read(seal_ref)
    require(seal['status']=='CLOSED_IMMUTABLE' and seal.get('fixture') is False and seal['assets'],'closed real base seal required')
    require(seal['model']=='qwen3-omni-30b-a3b-instruct' and all(r['bytes']>0 and len(r['sha256'])==64 and
        r['path'].startswith('/mnt/cephfs/xingchao/omni/models/qwen3-omni-30b-a3b-instruct/') for r in seal['assets']), 'base identity differs')
    # Reuse authenticated sealed hashes; do not reread tens of GB of weights.
    from fresh_adapter import scope
    with scope():
        from policy import bootstrap_assets
        policies=bootstrap_assets(seal_ref,root)
    current,reference=(read(policies[k])['version'] for k in ('current_ref','reference_ref'))
    finite=read(config['finite_capture_binding_ref']);overlay=read(finite['bounded_input_overlay_ref'])
    learning=copy.deepcopy(read(HERE/'learning.template.json'))
    from node_training_task import select
    learning=select(learning,spec.get('node_training_task'),mode=mode)
    system_ref=pin(config['system_policy_ref'])
    learning.update(version=learning['version']+'@finite-real-'+digest(identity)[:16],system_version=system_ref['sha256'])
    for key,version in [('behavior_policy_vector',current),('current_policy_vector',current),('reference_policy_vector',reference)]:
        learning['kernel'][key]={n:version for n in ('P','S1','S2','D')}
    learning['adapter_routing']['frozen_upstream_adapter_vector']=copy.deepcopy(learning['kernel']['current_policy_vector'])
    if mode=='finite_engineering':learning['phases']=learning['phases'][:1]
    phase=learning['phases'][0]
    expected=spec['node_training_task']['node'] if spec.get('node_training_task') else 'P'
    require(phase['target_nodes']==[expected] and phase['rollout_scope']=='node','configured first node phase differs')
    phase['frozen_upstream_release']=overlay['phase_frozen_upstream_ref']['sha256']
    if mode=='finite_engineering':learning['task_loss']['phases']={phase['id']:learning['task_loss']['phases'][phase['id']]}
    else:
        for item in learning['phases']:
            if item['rollout_scope']=='node':item['frozen_upstream_release']=overlay['phase_frozen_upstream_ref']['sha256']
    lr=write(root/'LEARNING.json',learning)
    from finite_runtime import data_module
    data=data_module(config)
    material_path=Path(data.__file__).parent/'prepare_sources.py'
    from dependencies import load
    materials=load(material_path,'finite_runtime_real_material').prepare(config['finite_capture_binding_ref'],root/'source-material')
    profile_value=copy.deepcopy(read(ROOT/'rollout-capture-contract/CONFIG.json')['profile'])
    if 'assistant_prefix' in spec:
        from dependencies import load as load_protocol
        protocol=load_protocol(ROOT/'rollout-capture-contract/capture.py','runtime_prompt_contract')
        profile_value['assistant_prefix']=spec['assistant_prefix']
        protocol.prefix_mode(profile_value)
    if 'output_contract' in spec:
        from dependencies import load as load_protocol
        protocol=load_protocol(ROOT/'rollout-capture-contract/capture.py','runtime_output_contract')
        profile_value['output_contract']=spec['output_contract']
        protocol.output_contract_mode(profile_value)
    profile=write(root/'PROFILE.json',profile_value)
    source_pins={n:pin(ROOT/'unified-learning'/n) for n in ('kernel.py','learning.py','dataset.py','scoring.py','schedule.py','fixtures.py')}
    source_pins.update({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    context_limit=spec.get('context_limit',read(HERE/'runtime.json')['model'].get('context_limit',8192))
    require(type(context_limit) is int and context_limit>0,'positive configured training context limit')
    consumer=dict(source_pins=source_pins,sealed_adapter_ref=pin(ROOT/'training-consumer-adapter/adapter.py'),
        context_limit=context_limit,pad_token_id=0,
        sampling_distribution=read(profile)['sampling'],recording_map_ref=materials['recording_map_ref'],learning_config_ref=lr)
    task=upstream()
    from scorer_extension import install
    install(task)
    clock=task.create_clock(learning,consumer_config=consumer)
    state_ref=write(root/'CLOCK.json',clock.state);consumer['clock_state_ref']=state_ref
    cr=write(root/'CONSUMER.json',consumer)
    authority=read(config['authorization_ref'])
    require(config['registry']['authority_ref']==pin(config['authorization_ref']),'Registry authority differs')
    require(config['registry']['cost_unit']=='logical_tokens_upper','Registry cost unit differs')
    for key in ('authority_ref','resources_ref'):pin(config['registry'][key])
    expires=datetime.datetime.fromisoformat(authority['expires_at']).timestamp()
    require(config['deadline_epoch']<=expires,'runtime deadline exceeds authority')
    require(config['budget']['total_tokens']<=authority['max_cost_units'],'runtime budget exceeds authority')
    require(config['budget']['prior_tokens_upper']+finite['bounds']['new_total_tokens_upper']<=config['budget']['total_tokens'],'finite budget insufficient')
    program=read(config['program_template_ref'])
    if 'manifest' not in program:program=read(spec['asset_program_ref'])
    manifest=read(program['manifest'])
    require(manifest['dataset']==pin(config['datasets']['TRAIN']),'program TRAIN source differs')
    business=bind_preparation_authority(pin(manifest['business']),root/'cpu-preparation')
    transport=write(root/'TRANSPORT.json',dict(read(spec['transport_config_ref']),max_cost_per_request=
        finite['bounds']['prompt_tokens_upper_per_call']+finite['bounds']['completion_tokens_upper_per_call']))
    prepared_cache_bytes=spec.get('prepared_cache_bytes',64*1024*1024)
    require(type(prepared_cache_bytes) is int and prepared_cache_bytes>=0,'nonnegative CPU prepared cache bytes per worker')
    source=dict(schema='configured-fresh-cohort-v1',cohort_id=config['run_id'],output_root=str(root/'cohort'),
        retained_source_material_ref=pin(root/'source-material/SOURCES.json'),
        native_task_root=str(root/'native'),progress_check_seconds=900,rollout_scope='node',
        screen_ref=overlay['screen_ref'],sampling=copy.deepcopy(overlay['sampling']),
        policy=dict(**policies,runtime_evidence_ref=None),learning_config_ref=lr,clock_state_ref=state_ref,consumer_config_ref=cr,
        capture=dict(enabled=False,seed=config['root_seed'],unknown='journal_reconcile_never_resend',retry=False,
            runtime_storage_budget_bytes=finite['bounds']['capture_storage_bytes'],profile_ref=profile,
            prepared_cache_bytes=prepared_cache_bytes,
            endpoint_policy_ref=None,business_config_ref=business,transport_config_ref=transport),
        native=dict(enabled=False,attempt_id=config['run_id'],model_path=read(HERE/'runtime.json')['model']['path'],
            actor_source_sha256=read(HERE/'runtime.json')['training']['actor_sha256'],
            max_successful_optimizer_steps=1,runtime_storage_budget_bytes=finite['bounds']['training_storage_bytes']),
        authorization=dict(authority,model_calls=False,native_training=False),execution_registry=copy.deepcopy(config['registry']),
        budget=dict(total_tokens=config['budget']['total_tokens'],cumulative_prior_tokens_upper=config['budget']['prior_tokens_upper']),
        resources=dict(rank=spec['rank']))
    if spec.get('node_difficulty') is not None:
        require(mode=='formal' and spec.get('node_training_task') and
                spec['node_difficulty']['node']==spec['node_training_task']['node'],
                'node difficulty must match the formal independent node task')
        source['sampling']['node_difficulty']=copy.deepcopy(spec['node_difficulty'])
    bind_official_source(source,root,spec.get('official_source_config_ref'))
    require(not source['policy'].get('fixture'),'synthetic input forbidden')
    return read(write(marker,dict(identity=identity,input_ref=write(root/'INPUT.json',source),learning_ref=lr,
        endpoint_probe_packet_ref=materials['endpoint_probe_packet_ref'],endpoint_probe_query_id=materials['endpoint_probe_query_id'],
        actual_model_calls=0,witness_required=True,lease_required=True,parity_required=True,training_ready=False)))
