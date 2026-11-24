"""Check real window scheduling and multi-process chosen-token scoring on CPU."""
import argparse,hashlib,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for name in ('cohort','config','release'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    sys.path[:0]=[str(args.release/'tree/parents/native-model-integration'),
        str(args.release/'tree/parents/jev-system-rl/implementation'),str(Path(__file__).parent/'source')]
    from reference_batch import collect_window,schedule,score_missing
    import os,torch,torch.distributed as dist
    from contextlib import nullcontext
    from types import SimpleNamespace
    from forward import forward
    cfg=json.loads(args.config.read_bytes())['reference_batch']
    inv=json.loads((args.cohort/'GENERATION-INVENTORY.json').read_bytes())
    assert inv['closed']
    groups=[]
    for name,ref in inv['files'].items():
        if len(Path(name).parts)!=3 or not name.startswith('groups/') or not name.endswith('/GENERATED.json'):continue
        raw=(args.cohort/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==ref['sha256']
        groups.append(json.loads(raw))
    def context(p):
        w=p['wire']
        return hashlib.sha256(json.dumps([w['messages'],w.get('chat_template_kwargs',{}),
            w.get('add_generation_prompt',True),w.get('continue_final_message',False)],sort_keys=True).encode()).hexdigest()
    lengths,items,requests=collect_window(groups,context)
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK'])
    plan=schedule(lengths,8,cfg['rows_per_rank'],cfg['padded_tokens_per_rank'])
    assert sorted(k for wave in plan for lane in wave for k in lane)==sorted(lengths)
    # Unequal prompt, action, audio widths; arbitrary tail rank. Check actual
    # multimodal packing and original forward/gather against per-row execution.
    class Model(torch.nn.Module):
        def forward(self,input_ids,attention_mask,position_ids,input_features,feature_attention_mask,**_):
            s=(input_ids*attention_mask).cumsum(-1)+position_ids[2]+(input_features[:,0,:]*feature_attention_mask).sum(-1).unsqueeze(1)
            logits=-((s.long()%11).unsqueeze(-1)-torch.arange(11)).abs().float()
            return SimpleNamespace(logits=logits)
    model=Model()
    actor=SimpleNamespace(actor_module=model,device_name='cpu',param_dtype=torch.bfloat16)
    def make(k):
        i=int(k);p=3+i%3;a=1+i%2;ids=(torch.arange(p+a)[None]+i)%11
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=(torch.arange(p+a)[None,None,:]+i).expand(1,3,-1),
            responses=ids[:,-a:],response_mask=torch.ones_like(ids[:,-a:]),multi_modal_inputs=[dict(
            input_features=torch.ones(1,2,i+2),feature_attention_mask=torch.ones(1,i+2,dtype=torch.long))])
    small={str(i):(3+i%3,1+i%2) for i in range(world*2+1)}
    dist.init_process_group('gloo')
    try:
        scores,stats=score_missing(SimpleNamespace(actor=actor),SimpleNamespace(disable_adapter=nullcontext),small,
            make,rows=2,tokens=32,pad_id=0)
        for k in small:
            _,value=forward(actor,make(k),temperature=1.,feature_dtype=torch.bfloat16)
            torch.testing.assert_close(torch.tensor(scores[k]),value[0],atol=0,rtol=0)
        assert model.training and stats['tail_participation_rows']==world-1
        import tempfile
        from reference_batch import frozen_reference_window
        from types import SimpleNamespace as Obj
        model.add_module('marker',torch.nn.Linear(1,1))
        sample=[]
        for j,k in enumerate([0,1,0,4]):
            row=make(str(k)); p=small[str(k)][0]
            sample.append(dict(packet=dict(stage='P',request_id=str(j),example=k),token_ids=row['responses'][0].tolist(),
                prompt_token_ids=row['input_ids'][0,:p].tolist(),behavior_logprobs=[-1.]*small[str(k)][1]))
        fixture=[dict(episodes=[dict(calls=sample)])]
        prepared=Obj(conditioning_key=lambda packet:str(packet['example']),prepare=lambda packet:dict(
            key=packet['example'],encoded={'input_ids':make(str(packet['example']))['input_ids'][:,:small[str(packet['example'])][0]]}))
        owner=Obj(actor=actor,tokenizer=Obj(pad_token_id=0))
        adapter=Obj(disable_adapter=nullcontext)
        def build(entry,*_):return make(str(entry['key']))
        directory=[tempfile.mkdtemp() if rank==0 else None];dist.broadcast_object_list(directory,src=0)
        root=Path(directory[0]);kwargs=dict(cache_root=root,namespace={'frozen':'test'})
        try:
            first,counts=frozen_reference_window(owner,adapter,fixture,prepared,build,cfg,**kwargs)
            second,warm=frozen_reference_window(owner,adapter,fixture,prepared,build,cfg,**kwargs)
            assert counts['unique_scored']==3 and warm['collective_forwards']==0
            assert all(first[k][0]==second[k][0] and second[k][1]['hit'] for k in first)
            assert first['0'][0]==first['2'][0] and first['2'][1]['source_request_id']=='0'
            _,changed=frozen_reference_window(owner,adapter,fixture,prepared,build,cfg,cache_root=root,namespace={'frozen':'different'})
            assert changed['unique_scored']==3
        finally:
            dist.barrier()
            if rank==0:
                import shutil
                shutil.rmtree(root)
        if rank==0:print(json.dumps(dict(real_requests=len(requests),unique=len(items),real_groups=len(groups),
            planned_fsdp8_waves=len(plan),source='real roster; numerical test uses CPU test model',test=stats,warm_reference_collectives=warm['collective_forwards'])))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
