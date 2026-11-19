"""Independent multi-device actual-operand NPU probes; no service mutations."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.pending')
    with tmp.open('x') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def initialize_device(c,device,torch):
    if c.get('probe_kind')=='expert_collective' or c.get('gate_collective',False) or c.get('down_collective',False) or c.get('grouped_family_collective',False):
        import re
        value=c['hccl_npu_socket_port_range']
        assert re.fullmatch(r'\d+-\d+',value),'explicit isolated HCCL port range'
        low,high=map(int,value.split('-'));assert 1024<=low<high<=65535
        os.environ['HCCL_NPU_SOCKET_PORT_RANGE']=value
    if c.get('custom_fused',False) or c.get('custom_op_bootstrap_before_device',False):
        assert not torch.npu.is_initialized(),'custom operator bootstrap requires a fresh NPU process'
        from vllm_ascend.utils import bootstrap_custom_op_env
        bootstrap_custom_op_env()
        assert os.environ.get('ASCEND_CUSTOM_OPP_PATH'),'custom op path must precede NPU initialization'
    torch.npu.set_device(device);torch.set_num_threads(2)
    if c.get('triton_device_properties',False):
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
        init_device_properties_triton()


def worker(c,device):
    import torch,torch_npu
    initialize_device(c,device,torch)
    if c.get('probe_kind')=='audio_conv_context':
        from audio_conv_context_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='rollout_sampler':
        from rollout_sampler_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='router_selection':
        from router_selection_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='down_layout':
        from down_layout_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='audio_family':
        from audio_family_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='projection_family':
        from projection_family_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='grouped_family':
        from grouped_family_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='grouped_vjp':
        from grouped_vjp_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='attention_contract':
        from attention_contract_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='block_random':
        from block_random_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='next_attention':
        from next_attention_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='next_projection':
        from next_projection_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='hf_moe_replay':
        from hf_moe_replay_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='expert_collective':
        from expert_collective_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='expert_tp':
        from expert_tp_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='expert_dispatch':
        from expert_dispatch_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='actual_expert':
        from actual_expert_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='expert':
        from expert_operator_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='router':
        from router_operator_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    if c.get('probe_kind')=='post_norm':
        from post_norm_probe import analyze
        save(Path(c['output'])/f'device-{device}.json',analyze(c,device))
        return
    root=Path(c['evidence']);folder=root/'hf/rank-0';m=json.loads((folder/'MANIFEST.json').read_text())
    assert m['complete'] and not m['failure']
    def load(name):
        e=m['events'][name];p=folder/e['file'];assert p.parent==folder and e['dtype']=='torch.bfloat16'
        data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==e['sha256']
        return torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(e['shape']).to('npu:'+str(device))
    x=load('attention');w=load('o_proj.base_layer.weight');a=load('o_proj.lora_A.default.weight');b=load('o_proj.lora_B.default.weight')
    expected=load('o_proj').reshape(4998,2048)
    def metric(left,right):
        l=left.float();r=right.float();d=(l-r).abs()
        return dict(elements=l.numel(),unequal=int((l!=r).sum()),max_abs=float(d.max()),finite=bool(torch.isfinite(r).all()))
    result=dict(device=device,torch=torch.__version__,torch_npu=torch_npu.__version__,
                source=str(folder/'MANIFEST.json'),source_sha256=hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest(),
                training=False,production_acceptance=False,records=[])
    if c.get('gradient_probe',False):
        from row_gradient_probe import analyze
        result['training']=True
        for rows in c['rows_by_device'][str(device)]:
            result['records'].append(analyze(torch,x,w,a,b,rows,c['gradient_pad_rows']))
        result['peak_allocated_bytes']=torch.npu.max_memory_allocated()
        save(Path(c['output'])/f'device-{device}.json',result)
        return
    with torch.no_grad():
        full_x=x.unsqueeze(0)
        ref_base=torch.nn.functional.linear(full_x,w)
        ref_u=torch.nn.functional.linear(full_x,a)
        ref_z=torch.nn.functional.linear(ref_u,b)
        ref_result=ref_base+ref_z*2
        result['same_device_full_vs_captured_HF']=metric(expected,ref_result.reshape(4998,2048))
        for n in c['rows_by_device'][str(device)]:
            xx=x[:n].unsqueeze(0)
            torch.npu.synchronize();start=time.monotonic()
            base=torch.nn.functional.linear(xx,w)
            u=torch.nn.functional.linear(xx,a);z=torch.nn.functional.linear(u,b)
            full=base+z*2
            parts=[torch.nn.functional.linear(xx[...,i*1024:(i+1)*1024].contiguous(),w[:,i*1024:(i+1)*1024].contiguous()) for i in range(4)]
            partbase=parts[0].float()
            for t in parts[1:]:partbase=partbase+t.float()
            ua=torch.nn.functional.linear(xx.float(),a.float())
            za=torch.nn.functional.linear(ua,b.float()*2)
            fp32_lora=(base.float()+za).bfloat16()
            padded_checks={}
            for size in c.get('pad_rows',[]):
                assert size>=n and size<=4998
                padded=torch.zeros((1,size,4096),dtype=x.dtype,device=x.device)
                padded[:,:n].copy_(xx)
                pb=torch.nn.functional.linear(padded,w)[:,:n]
                pu=torch.nn.functional.linear(padded,a)
                pz=torch.nn.functional.linear(pu,b)[:,:n]
                padded_checks[str(size)]=dict(base_vs_full=metric(ref_base[:,:n],pb),
                                             result_vs_full=metric(ref_result[:,:n],pb+pz*2))
                del padded,pb,pu,pz
            torch.npu.synchronize()
            result['records'].append(dict(rows=n,seconds=time.monotonic()-start,
                full_vs_captured_HF=metric(expected[:n],full.reshape(n,2048)),
                same_device_fullwidth_stages={
                    'base':metric(ref_base[:,:n],base),'u':metric(ref_u[:,:n],u),
                    'z':metric(ref_z[:,:n],z),'result':metric(ref_result[:,:n],full)},
                partition_base_vs_full=metric(base,partbase.bfloat16()),
                FP32_lora_vs_HF_consumption=metric(full,fp32_lora),
                padded_checks=padded_checks,
                peak_allocated_bytes=torch.npu.max_memory_allocated(),
                limitation='independent real NPU linear calls; no TP communication or whole-model proof'))
            del base,u,z,full,parts,partbase,ua,za,fp32_lora
    save(Path(c['output'])/f'device-{device}.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',type=int)
    args=p.parse_args();c=json.loads(Path(args.config).read_text())
    if args.device is not None:return worker(c,args.device)
    out=Path(c['output']);out.mkdir(exist_ok=False)
    fs=subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(out)],text=True).strip()
    assert fs in ['ext4','xfs','btrfs'],'local durable output required'
    assert set(c['devices'])=={0,1,2,3} and c['max_parallel']==4
    children=[];start=time.time()
    for d in c['devices']:
        log=(out/f'device-{d}.log').open('xb')
        child=subprocess.Popen([sys.executable,'-B',__file__,'--config',args.config,'--device',str(d)],stdout=log,stderr=subprocess.STDOUT)
        children.append((d,child,log))
    save(out/'START.json',dict(started=start,children=[dict(device=d,pid=p.pid) for d,p,l in children],config=c))
    codes={}
    for d,child,log in children:
        try:codes[str(d)]=child.wait(timeout=max(1,c['timeout_seconds']-(time.time()-start)))
        except subprocess.TimeoutExpired:
            child.terminate();codes[str(d)]=child.wait(timeout=30)
        finally:log.close()
    save(out/'COMPLETE.json',dict(exit_codes=codes,passed=all(v==0 for v in codes.values()),seconds=time.time()-start,production_acceptance=False))
    raise SystemExit(0 if all(v==0 for v in codes.values()) else 1)


if __name__=='__main__':main()
