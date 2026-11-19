"""Execute the retained audio tower and embedding merge on all real TP ranks."""
import base64
import hashlib
import json
import os
from pathlib import Path


def replay(worker, config):
    import torch
    from safetensors.torch import save_file
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from family_precision import install
    from run_package import read

    root = Path(config['input'])
    def load(key):
        ref = config[key]
        data = (root/ref['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['sha256']:
            raise ValueError('retained input identity mismatch: '+key)
        return json.loads(data)

    with set_current_vllm_config(worker.vllm_config):
        receipt = install(worker, read(config['precision_ref']))
        model = worker.model_runner.get_model()
        tower = model.audio_tower
        feat = load('features')
        buf = base64.b64decode(feat['bf16'], validate=True)
        assert hashlib.sha256(buf).hexdigest() == feat['sha256']
        f = torch.frombuffer(bytearray(buf),dtype=torch.bfloat16).reshape(feat['shape']).to(tower.device)
        prompt_ids = torch.tensor(load('behavior')['records'][config['action_index']]['prompt_token_ids'],device='cpu')
        ids = prompt_ids.to(tower.device)
        # Match GPUModelRunner._gather_mm_embeddings: the model combines this
        # mask with CPU token comparisons before scattering the audio rows.
        where = prompt_ids.eq(model.config.audio_token_id)
        row_audit, row_restore = {}, None
        if 'audio_row_sites' in config:
            from retained_audio_row_precision import attach
            row_audit,row_restore=attach(tower,config['audio_row_sites'],worker.rank)
        audio_attention_restore = None
        if 'audio_attention_layers' in config:
            from retained_audio_attention_precision import attach
            audio_attention_restore = attach(tower,config['audio_attention_layers'],worker.rank,
                                             compare_native=True)
        inspect_layer = config.get('inspect_layer')
        finish_layer = None
        if inspect_layer is not None:
            from retained_audio_layer import observe
            finish_layer = observe(tower.layers[inspect_layer], 'vl', inspect_layer, worker.rank,
                                   Path(config['output'])/f'rank-{worker.rank}')
        seen = {}
        def capture(key,tensor):
            if key in seen: raise ValueError('multiple calls to '+key)
            seen[key] = tensor.detach().contiguous().cpu()
        hooks = [tower.layers[0].register_forward_pre_hook(lambda m,args:capture('layer.0.in',args[0]))]
        for i,layer in enumerate(tower.layers):
            hooks.append(layer.register_forward_hook(lambda m,args,out,i=i:capture(f'layer.{i}.out',out)))
        try:
            with torch.inference_mode(), set_forward_context(None, worker.vllm_config):
                audio = model._process_audio_input(dict(input_features=f,audio_feature_lengths=torch.tensor([f.shape[-1]])))
                assert len(audio)==1 and audio[0].shape[0]==int(where.sum())
                capture('audio_out',audio[0])
                capture('merged_prompt',model.embed_input_ids(ids,audio,is_multimodal=where))
        finally:
            for hook in hooks:hook.remove()
            if row_restore:row_restore()
            audio_attention_receipt = audio_attention_restore() if audio_attention_restore else None
        assert len(seen)==len(tower.layers)+3 and all(torch.isfinite(x).all() for x in seen.values())
        assert torch.equal(seen['merged_prompt'][where.cpu()],seen['audio_out'])
        out = Path(config['output'])/f'rank-{worker.rank}'
        out.mkdir()
        target = out/'VL.safetensors'
        save_file(seen,str(target))
        with target.open('rb') as stream:os.fsync(stream.fileno())
        layer_ref = {'layer_sha256':finish_layer()} if finish_layer else {}
        assert all(x==1 for x in row_audit.values())
        return dict(**layer_ref,audio_row_calls=row_audit,audio_attention=audio_attention_receipt,
                    rank=worker.rank, precision=receipt, layers=len(tower.layers),audio_shape=list(audio[0].shape),
                    merged_shape=list(seen['merged_prompt'].shape), sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def main(path):
    import sys
    from dependencies import HERE, ROOT
    cfg=json.loads(Path(path).read_bytes())
    parent=ROOT/'policy-iteration-successor'
    sys.path.insert(0,str(parent))
    from dependencies import load
    bootstrap=load(HERE/'service_bootstrap.py','retained_original_bootstrap')
    worker_process_environment=bootstrap.worker_process_environment
    custom_operator_environment=bootstrap.custom_operator_environment
    worker_process_environment()
    custom_operator_environment()
    from sync_overlay import verify
    overlay=verify(cfg['named_rpc_overlay'])
    acl=ROOT/'native-recompute-integration/state-repair/acl-runtime-successor'
    # Spawn workers use the same explicitly named, JSON-only RPC mechanism as
    # production. Extend only the scoped worker module in the local run.
    rpc=Path(cfg['output'])/'boundary-rpc'
    rpc.mkdir()
    for src in cfg['named_rpc_overlay']['files'].values():
        source=Path(src['path'])
        payload=source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=src['sha256']:
            raise ValueError('source changed while installing named RPC')
        if source.name=='ascend_fp32_worker.py':
            payload+=b'\nfrom retained_audio_vllm_boundary import replay\nFP32LoRAWorkerExtension.omni_retained_audio_boundary = replay\n'
        if source.name=='ascend_fp32_worker.py' and 'retained_actions' in cfg:
            payload+=b'\nfrom retained_action_replay import install_rows,close_rows\nFP32LoRAWorkerExtension.omni_retained_install_rows=install_rows\nFP32LoRAWorkerExtension.omni_retained_close_rows=close_rows\n'
        if source.name=='ascend_fp32_worker.py' and ('trace_predictor' in cfg or 'fresh_trace' in cfg):
            payload+=b'\nfrom retained_decode_rows import begin,finish\nFP32LoRAWorkerExtension.omni_predictor_begin=begin\nFP32LoRAWorkerExtension.omni_predictor_finish=finish\n'
        with (rpc/source.name).open('xb') as f:
            f.write(payload);f.flush();os.fsync(f.fileno())
    search=[str(rpc),overlay,str(acl),str(HERE)]
    sys.path[:0]=search
    os.environ['PYTHONPATH']=':'.join(search+[os.environ.get('PYTHONPATH','')])
    from engine_api import build_engine
    engine=build_engine(cfg['engine'])
    rows=engine.collective_rpc('omni_retained_audio_boundary', args=(cfg,),timeout=240)
    assert sorted(row['rank'] for row in rows)==[0,1,2,3]
    result=dict(ranks=rows, model=cfg['engine']['model'], actual_tp=4, language_layers_executed=0,
                sampled_actions=0, production_acceptance=False)
    if 'retained_actions' in cfg:
        from dispatch_retained_hf_boundary import save
        save(Path(cfg['output'])/'MODULE-RESULT.json',result)
        from retained_action_replay import decode
        installed=engine.collective_rpc('omni_retained_install_rows',
            args=(cfg['audio_row_sites'],cfg.get('audio_control'),cfg.get('gate_candidate_layers'),
                  cfg.get('attention_candidate_layers'),cfg.get('down_candidate_layers'),
                  cfg.get('audio_attention_layers'),*([cfg.get('router_layers'),cfg['kv_layers']] +([cfg['tiled_layers']] if 'tiled_layers' in cfg else []) if 'kv_layers' in cfg else [cfg['router_layers']] if 'router_layers' in cfg else [])),timeout=240)
        assert sorted(installed)==list(range(4))
        try:
            if 'fresh_group' in cfg:
                from retained_group_parity import sample
                result['fresh_group']=sample(engine,cfg)
                result['retained_decode']=[]
            else:
                result['retained_decode']=decode(engine,cfg)
            result['language_layers_executed']=48
            result['sampled_actions']=sum(c['generated']+c.get('forced_prefill',{}).get('extra_generated',0)
                                         +c.get('cache_pair',{}).get('generated',0) for c in result['retained_decode'])
            if 'fresh_group' in result:
                result['sampled_actions']=sum(len(r['token_ids']) for r in result['fresh_group']['records'])
        finally:
            result['decode_row_receipts']=engine.collective_rpc('omni_retained_close_rows',timeout=240)
    with (Path(cfg['output'])/'RESULT.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
