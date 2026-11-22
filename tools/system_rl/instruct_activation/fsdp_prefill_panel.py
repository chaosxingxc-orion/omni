"""No-update scorer for a complete fixed-batch panel on the restored training worker."""
import hashlib
import json
from pathlib import Path


def evaluate(worker, adapted, spec, sealed, out, rank, world, checkpoint_ref, checkpoint):
    from dependencies import require
    from storage import commit
    from retained_group_parity import hf_scores
    from lora_row_precision import attach_hf

    path = sealed(spec['updated_http_parity'])
    packet = json.loads(path.read_bytes())
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = packet['rows']
    adapter = checkpoint['ranks'][0]['files']['lora_adapter/adapter_model.safetensors']
    require(packet['schema']=='natural-prefill-score-panel-v1' and packet['training_use'] is False
            and len(rows)>0 and len(rows)%world==0 and packet['width']==world==checkpoint['world_size']
            and packet['adapter_sha']==adapter and packet['checkpoint_sha']==checkpoint_ref['sha256'],
            'training checkpoint, topology or natural panel mismatch')
    inputs, outputs = [], []
    for i,row in enumerate(rows):
        p = row['packet']
        inputs.append(dict(group=i,replicate=0,packet=p,prompt_token_ids=row['prompt'],source_sha256=manifest_hash))
        outputs.append(dict(group=i,replicate=0,request_id=p['request_id'],wire_sha256=p['wire_sha256'],
                            seed=p['wire']['seed'],source_sha256=manifest_hash,token_ids=row['tokens'],
                            behavior_logprobs=row['logprobs'],raw_top1=None))
    worker.actor.actor_module.eval()
    close=attach_hf(adapted.get_base_model().model,spec['lora_row_geometry'])
    try:
        observed=hf_scores(worker,adapted,dict(schema='bounded-train-natural-requests-v1',rows=inputs),
                           dict(manifest_sha256=manifest_hash,records=outputs),manifest_hash,rank,world,
                           actor_forward=True)
    except BaseException:
        close(check_coverage=False)
        raise
    geometry=close()
    result=[]
    for i,s in zip(range(rank,len(rows),world),observed,strict=True):
        require(s['index']==i and s['request_id']==inputs[i]['packet']['request_id'] and
                all(len(s[k])==len(rows[i]['tokens']) for k in ('scores','hf_top1','train_scores','train_top1')),
                'missing train or eval actions')
        result.append(dict(index=i,response=rows[i]['response'],**{k:s[k] for k in
                           ('scores','hf_top1','train_scores','train_top1')}))
    commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,rows=result,panel=manifest_hash,
           checkpoint=checkpoint_ref['sha256'],adapter=adapter,step=checkpoint['step'],optimizer_updates=0,
           lora_row=geometry,arithmetic=worker.close_actor_arithmetic()))
