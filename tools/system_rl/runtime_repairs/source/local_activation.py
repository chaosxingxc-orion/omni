"""Bind the original policy overlay to a guarded, local TP4 activation lane."""
import fcntl
import os
import sys
from pathlib import Path
from io_utils import doc, pin, raw, require, save

ROW_ARGS = ('audio_row_sites', 'gate_candidate_layers', 'attention_candidate_layers',
            'down_candidate_layers', 'audio_attention_layers', 'router_layers',
            'kv_layers', 'tiled_layers')


def owner_check(config):
    require(config['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'ROLLOUT_BOOT_CHANGED')
    # A separate open file description must be unable to acquire a shared
    # lock. The original local launcher/controller owns the exclusive guard.
    with Path(config['guard']).open('rb') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    raise ValueError('ORIGINAL_ROLLOUT_EXCLUSIVE_GUARD_REQUIRED')


def prepare(config, rpc_root):
    from dependencies import register_locations
    locations=doc(config['locations_ref'])
    register_locations({ref['logical']:ref['path'] for ref in locations})
    for ref in locations:
        raw(ref)
    root=str(Path(rpc_root).parent/'arithmetic')
    sys.path.insert(0,root)
    os.environ['PYTHONPATH']=root+':'+os.environ['PYTHONPATH']


def is_decode(rows):
    from vllm.forward_context import get_forward_context
    meta = get_forward_context().attn_metadata
    if not isinstance(meta, dict) or not meta:
        return False
    entry = next(iter(meta.values()))
    return (str(entry.attn_state).split('.')[-1] == 'DecodeOnly'
            and list(entry.actual_seq_lengths_q) == list(range(1, rows + 1)))


def install_rows(worker, rows):
    from dependencies import register_locations
    from retained_action_replay import install_rows as attach
    from vllm.config import set_current_vllm_config
    register_locations({ref['logical']:ref['path'] for ref in rows['locations']})
    config=rows['precision']
    with set_current_vllm_config(worker.vllm_config):
        precision=worker.omni_production_precision(config)
        row_config=rows['rows']
        rank=attach(worker,row_config['audio_row_sites'],None,
                    **{key:row_config[key] for key in ROW_ARGS[1:]})
        grouped={}
        for site in ('q','o'):
            layers=row_config.get(site+'_base_layers',[])
            if layers:
                from retained_kv_candidate import attach as attach_base
                text=worker.model_runner.get_model().language_model.model
                setattr(worker,'_omni_'+site+'_base',attach_base(text,layers,side='vl',projections=(site+'_proj',)))
                grouped[site+'_grouped_layers']=layers
        if 'frozen_projection_cache' in row_config:
            import hashlib, frozen_projection_cache
            settings = dict(row_config['frozen_projection_cache'])
            require(hashlib.sha256(Path(frozen_projection_cache.__file__).read_bytes()).hexdigest() == settings.pop('source_sha256'),
                    'frozen projection cache source changed')
            grouped['frozen_projection_cache'] = frozen_projection_cache.attach(worker,settings)
        if 'decode_projection_runtime' in row_config:
            import hashlib, decode_projection_runtime
            require(hashlib.sha256(Path(decode_projection_runtime.__file__).read_bytes()).hexdigest()
                    == row_config['decode_projection_runtime'], 'direct projection implementation changed')
            from family_precision import ATTR
            family = getattr(worker.model_runner.get_model(), ATTR)['config']
            decode_projection_runtime.attach(worker,family)
            grouped['decode_projection_runtime'] = row_config['decode_projection_runtime']
        if 'lora_row_geometry' in row_config:
            import hashlib,lora_row_precision
            spec=dict(row_config['lora_row_geometry'])
            require(hashlib.sha256(Path(lora_row_precision.__file__).read_bytes()).hexdigest()==spec.pop('source_sha256'),
                    'LoRA row precision source changed')
            worker._omni_lora_geometry=lora_row_precision.attach(worker.model_runner.get_model().language_model.model,spec)
            grouped['lora_row_geometry']=row_config['lora_row_geometry']
        if 'decode_expert_owners' in row_config:
            import hashlib, decode_expert_owners
            settings=dict(row_config['decode_expert_owners'])
            require(hashlib.sha256(Path(decode_expert_owners.__file__).read_bytes()).hexdigest()==settings.pop('source_sha256'),
                    'expert owner implementation changed')
            grouped['decode_expert_owners']=decode_expert_owners.attach(worker,settings)
        if 'decode_layer_graph' in row_config:
            import hashlib, decode_layer_graph
            settings=dict(row_config['decode_layer_graph'])
            require(hashlib.sha256(Path(decode_layer_graph.__file__).read_bytes()).hexdigest()==settings.pop('source_sha256'),
                    'decode graph implementation changed')
            decode_layer_graph.attach(worker,settings)
            grouped['decode_layer_graph']=settings
    return dict(rank=rank, family=precision, **grouped)


async def install_precision(engine, server, local):
    from service_overlay import awaited
    spec_ref=server['named_rpc_overlay']['precision_ref']
    rows=doc(local['rows_ref'])
    require(set(rows)-{'q_base_layers','o_base_layers','lora_row_geometry','frozen_projection_cache','decode_expert_owners','decode_projection_runtime','decode_layer_graph'}==set(ROW_ARGS),'expected complete retained precision composition')
    result=await awaited(engine.collective_rpc('omni_local_install_rows',timeout=240,args=(dict(
        precision=doc(spec_ref), locations=doc(local['locations_ref']), rows=rows),)))
    require(sorted(r['rank'] for r in result)==list(range(4)) and
            all(r['family']==result[0]['family'] and r.get('lora_row_geometry')==rows.get('lora_row_geometry') and all(r.get(k+'_grouped_layers',[])==rows.get(k+'_base_layers',[]) for k in ('q','o'))
                for r in result), 'FOUR_TP_PRECISION_WORKERS')
    if 'decode_expert_owners' in rows:
        require(all(r.get('decode_expert_owners')==dict(layers=48,experts_per_rank_per_layer=32,
                    bytes=48*(32+int(rows['decode_expert_owners'].get('static_routes',False)))*2*(2048*1536+768*2048)) for r in result),'all TP4 expert owners required')
    if 'frozen_projection_cache' in rows:
        require(all(r.get('frozen_projection_cache')==dict(matrices=192,bytes=48*(1280*2048+2048*1024)*4*2) for r in result),
                'all four workers must retain the complete frozen projection cache')
    save(Path(server['overlay_root'])/'PRECISION.json',dict(specification_ref=spec_ref,
        rows_ref=local['rows_ref'],workers=result,actual_device_execution=True,
        updated_parity_required=True))
