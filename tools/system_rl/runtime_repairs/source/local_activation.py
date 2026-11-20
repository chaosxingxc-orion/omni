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
    return dict(rank=rank, family=precision, **grouped)


async def install_precision(engine, server, local):
    from service_overlay import awaited
    spec_ref=server['named_rpc_overlay']['precision_ref']
    rows=doc(local['rows_ref'])
    require(set(rows)-{'q_base_layers','o_base_layers'}==set(ROW_ARGS),'expected complete retained precision composition')
    result=await awaited(engine.collective_rpc('omni_local_install_rows',timeout=240,args=(dict(
        precision=doc(spec_ref), locations=doc(local['locations_ref']), rows=rows),)))
    require(sorted(r['rank'] for r in result)==list(range(4)) and
            all(r['family']==result[0]['family'] and all(r.get(k+'_grouped_layers',[])==rows.get(k+'_base_layers',[]) for k in ('q','o'))
                for r in result), 'FOUR_TP_PRECISION_WORKERS')
    save(Path(server['overlay_root'])/'PRECISION.json',dict(specification_ref=spec_ref,
        rows_ref=local['rows_ref'],workers=result,actual_device_execution=True,
        updated_parity_required=True))
