"""Compare independently produced recording waves with closed actor fields and schedules."""
import argparse,json,os,sys,time
from pathlib import Path
from threading import Event
from microbatch_producer import Producer
from microbatch_roster import group_rewards
from microbatch_fields import overlap_microbatches,CommitGate

def check(root,source,expected,native=False,acceptance=False):
    import torch
    from bounded_actor_step import generation_sources
    from dependencies import load,require
    from check_accepted_actor_packet import equal
    spec=json.loads(Path('/sealed/TRAIN.local.json').read_bytes())
    with generation_sources(spec) as rt:
        import fresh_adapter
        os.environ.update(OMNI_CAPTURE_CONSUMER_SOURCE=rt['consumer_source'],OMNI_CAPTURE_CONSUMER_WRAPPER=rt['consumer_wrapper'])
        with fresh_adapter.scope():
            from cohort_io import read,pin
            closed=load(rt['cohort_source'],'cohort')
            capture=load(rt['capture_source'],'wave_capture')
            cons=fresh_adapter.consumer(load(rt['consumer_wrapper'],'wave_consumer'))
            cfg=read(pin(root/'LOCAL-CONSUMER.json'));lc=read(cfg['learning_config_ref'])
            clock=read(cfg['clock_state_ref']);plan=read(cfg['selected_cohort_ref'])
            maps=read(cfg['recording_map_ref']);origins={o['recording_ref_id']:o for o in maps['records']}
            snap=clock['groups'][plan['group_ids'][0]]['snapshot']
            require(all(clock['groups'][g]['snapshot']==snap for g in plan['group_ids']),'mixed snapshot')
            first_done=Event();second_ready=Event();opened=[];pack_times=[]
            seq=lambda d:root/'groups'/str(d['draw_index'])
            def group(d):
                opened.append(d['group_id'])
                return closed.group_storage(read(pin(seq(d)/'GROUP.json')),expand=True)
            with cons.unified(cfg) as (learning,schedule,kernel):
                env=learning.evaluate.__globals__
                rewards=lambda d,g:group_rewards(d,g,snap,lc,labels_for=capture.labels_for,
                    score_episode=env['score_episode'],reward=learning.reward,advantages=env['reduce_validated'].__globals__['advantages'])
                started=time.monotonic()
                producer=Producer(plan,(read(pin(seq(d)/'GENERATED.json')) for d in plan['draws']),snap,
                    world=len(spec['devices']),micro=spec['actor_batch']['microbatch_per_card'],accumulation=spec['actor_batch']['gradient_accumulation_steps'],
                    group_size=lc['kernel']['G'],source=source,expected=expected,pad=cfg['pad_token_id'],context_limit=cfg['context_limit'],
                    reward_rows=rewards,origins=origins,group_reader=group)
                setup=time.monotonic()-started
            def pack(item):
                idx,w=item;expected_groups={producer.keys[i][0] for slots in w for i,owned in slots}
                if idx==1:
                    second_ready.set()
                    if not first_done.wait(30):raise ValueError('second wave not released')
                before=time.monotonic();prior=set(opened);v=producer.wave(w)
                require(set(opened)==(prior|expected_groups) and len(opened)==len(set(opened)),'read unrelated or duplicated group')
                pack_times.append(time.monotonic()-before)
                return v
            # Reference outputs enter the verifier only, never the producer.
            rec=read(pin(root/'actor-packet/RECEIPT.json'));payload=pin(root/'actor-packet/fields.pt')
            require(rec['payload']=={k:payload[k] for k in rec['payload']},'packet hash changed')
            packet=torch.load(root/'actor-packet/fields.pt',mmap=True,weights_only=True,map_location='cpu')
            gates=[CommitGate([w[r] for w in producer.layout['waves']],
                      dict(global_episode_denominator=producer.layout['global_episode_denominator'],snapshot=snap,
                           group_size=lc['kernel']['G'],global_query_count=len(plan['draws'])))
                   for r in range(len(spec['devices']))]
            import subprocess
            input_ready=[];gate_cost=0.;start=time.monotonic()
            child=None
            if acceptance:
                child=subprocess.Popen([sys.executable,'-B',__file__,'--only-accept','--root',str(root)],stdout=subprocess.PIPE,text=True)
            checked=0;seen=[]
            for idx,ranks in enumerate(overlap_microbatches(enumerate(producer.layout['waves']),pack)):
                if idx==0 and len(producer.layout['waves'])>1:
                    require(second_ready.wait(30),'second producer did not overlap consumption')
                try:
                    seen.append(len(opened));input_ready.append(time.monotonic()-start)
                    for rank,(slots,tensors,non) in enumerate(ranks):
                        gates[rank].observe(slots,tensors,non)
                        ids=[i for i,owned in slots];actual=read(pin(root.parent/f'rank-{rank}'/'LOSS.json'))['result']['events']
                        offset=idx*len(slots)
                        require([(e['row'],e['owned']) for e in actual[offset:offset+len(slots)]]==slots,'wrong microbatch')
                        equal(tensors,{k:v[ids] for k,v in packet['tensors'].items()})
                        equal({k:list(v) for k,v in non.items()},{k:[v[i] for i in ids] for k,v in packet['non_tensors'].items()})
                        checked+=sum(owned for _,owned in slots)
                finally:first_done.set()
            require(checked==len(producer.keys) and checked>0,'incomplete replay')
            for rank,gate in enumerate(gates):
                actual=read(pin(root.parent/f'rank-{rank}'/'LOSS.json'))['result']['events']
                width=spec['actor_batch']['microbatch_per_card']
                slots=[(e['row'],e['owned']) for e in actual];waves=[slots[i:i+width] for i in range(0,len(slots),width)]
                if rank==0:
                    # A wrong probability in the final acceptance must stop even
                    # after all provisional microbatches have been consumed.
                    altered=dict(packet,tensors=dict(packet['tensors']))
                    name='ref_log_prob';altered['tensors'][name]=packet['tensors'][name].clone()
                    altered['tensors'][name][slots[0][0],0]-=1.
                    try:gate.reconcile(altered,waves)
                    except ValueError as err:
                        if str(err)!='provisional fields differ from accepted actor fields':raise
                    else:raise ValueError('corrupt probability accepted')
                before=time.monotonic()
                gate.reconcile(packet,waves)
                gate_cost+=time.monotonic()-before
            admission=None
            if child is not None:
                out,_=child.communicate(timeout=180)
                require(child.returncode==0,'full CPU acceptance failed')
                admission=json.loads(out)
            comparison=None
            if native:
                from check_native_wave_loss import compare
                producer.groups.clear()
                comparison=compare(producer,packet,root)
            return dict(full_acceptance=admission,first_input_ready_seconds=input_ready[0],reconcile_all_ranks_seconds=gate_cost,native_comparison=comparison,all_ranks_reconciled=True,wrong_reference_rejected=True,rows=checked,recordings=len(plan['draws']),groups_read_at_wave=seen,producer_setup_seconds=setup,
                        wave_pack_seconds=pack_times,fields_exact=True,whole_batch_report_input=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',action='append',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--native',action='store_true');p.add_argument('--only-accept',action='store_true');p.add_argument('--acceptance',action='store_true');p.add_argument('--sha256');a=p.parse_args();sys.path.insert(1,'/sealed')
    if a.only_accept:
        from bounded_actor_step import generation_sources
        from dependencies import load
        with generation_sources(json.loads(Path('/sealed/TRAIN.local.json').read_bytes())) as rt:
            import fresh_adapter
            os.environ.update(OMNI_CAPTURE_CONSUMER_SOURCE=rt['consumer_source'],OMNI_CAPTURE_CONSUMER_WRAPPER=rt['consumer_wrapper'])
            with fresh_adapter.scope():
                from cohort_io import read,pin
                load(rt['cohort_source'],'cohort')
                cons=fresh_adapter.consumer(load(rt['consumer_wrapper'],'concurrent_acceptance'))
                r,=a.root;cfg=read(pin(r/'LOCAL-CONSUMER.json'));files=read(pin(r/'FILES.json'))
                start=time.monotonic();_,report,_=cons.build_with_bundle(files,cfg);elapsed=time.monotonic()-start
                if json.dumps(report,sort_keys=True)!=json.dumps(read(pin(r/'CONSUMER-REPORT.json')),sort_keys=True):raise ValueError('different report')
                print(json.dumps(dict(original_report_equal=True,seconds=elapsed)))
    else:
        if not a.source or not a.sha256:p.error('source and sha256 required')
        print(json.dumps([check(r,a.source,a.sha256,a.native,a.acceptance) for r in a.root]),flush=True)
