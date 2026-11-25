"""Replay the real asynchronous publisher, first graph lifetime and original input comparison on CPU."""
import argparse,copy,hashlib,json,os,shutil,sys,tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def replay(root):
    import numpy as np
    import torch
    from dependencies import load,require
    from run_package import read,pin
    from bounded_actor_step import generation_sources
    from first_wave_forward import FirstWave
    from pack_actor_fields import load_packet
    from check_accepted_actor_packet import equal
    cmd=read(root.parent/'COMMAND-INTENT.json')['command'];spec=read('/sealed/TRAIN.local.json')
    spec['generation']['generation_inventory_ref']=cmd['generation_inventory_ref']
    spec['generation']['pending_capture_ref']=cmd['pending_capture_ref']
    world=len(spec['devices']);rank=0
    historical=load_packet(root,cmd['generation_inventory_ref'])
    actual=read(root.parent/'rank-0/LOSS.json')['result']['layout']
    ids=[e['row'] for e in actual['schedules'][rank][:spec['actor_batch']['microbatch_per_card']]]
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.25));self.drop=torch.nn.Dropout(.1)
        def forward(self,ins):
            return ins['old_log_probs']-self.drop(self.scale*torch.ones_like(ins['old_log_probs'])).square()
    model=Head()
    worker=SimpleNamespace(actor=SimpleNamespace(actor_module=model))
    rng=torch.get_rng_state()
    def baseline(t):
        value=model(t);value.sum().backward()
        return value.detach().clone(),model.scale.grad.clone(),torch.get_rng_state()
    original=baseline({k:v[ids] for k,v in historical['tensors'].items()});model.zero_grad();torch.set_rng_state(rng)
    with tempfile.TemporaryDirectory(dir=Path(os.environ['TMPDIR']),delete=False) as tmp, generation_sources(spec) as rt, ExitStack() as life:
        from fresh_adapter import scope
        dst=Path(tmp)
        for item in ('SELECTED.json','RESERVED-CLOCK.json','LOCAL-CONSUMER.json'):
            shutil.copyfile(root/item,dst/item)
        for source in (root/'groups').glob('*'):
            target=dst/'groups'/source.name;target.mkdir(parents=True)
            for name in ('GROUP.json','GENERATED.json'):shutil.copyfile(source/name,target/name)
        os.environ.update(OMNI_CAPTURE_CONSUMER_SOURCE=rt['consumer_source'],OMNI_CAPTURE_CONSUMER_WRAPPER=rt['consumer_wrapper'])
        with scope():
            from cohort_io import read as rd,pin as pn
            load(rt['cohort_source'],'cohort');capture=load(rt['capture_source'],'capture_for_first_wave')
            cfg=rd(pn(root/'INPUT.json'));cfg['capture']['official_source_config_ref']=pn(root/'LOCAL-OFFICIAL-SOURCE.json')
            conf=rd(pn(root/'LOCAL-CONSUMER.json'));plan=rd(conf['selected_cohort_ref']);clock=rd(conf['clock_state_ref']);learn=rd(conf['learning_config_ref'])
            origins={o['recording_ref_id']:o for o in rd(conf['recording_map_ref'])['records']}
            geom=[]
            early=FirstWave(spec,rt,rank,world,worker,lambda:geom.append(True))
            with patch('resident.collectively',lambda fn:fn()),patch('bounded_actor_step.compact_scores',lambda actor,inputs:actor.actor_module(inputs)):
                early.overlap(dst,cfg,conf,plan,clock,learn,origins,capture.labels_for,
                    {'dp8_reserved_logical_upper':rd(pn(root/'NATIVE.accepted.json'))['dp8_reserved_logical_upper']},
                    pn(root/'LOCAL-CONSUMER.json'),rt['row_encoding'])
                packet=load_packet(dst,cmd['generation_inventory_ref'])
                equal(packet['tensors'],historical['tensors']);equal(packet['non_tensors'],historical['non_tensors']);equal(packet['meta'],historical['meta'])
                require(geom==[True] and early.scores is not None,'first graph did not survive publication')
                batch=SimpleNamespace(base=SimpleNamespace(rows=len(packet['non_tensors']['record_identity']),identities=packet['non_tensors']['record_identity']))
                with patch('events.partition',return_value=actual):early.bind(packet,None,batch)
                out=early.consume({'record_identity':[batch.base.identities[i] for i in ids]});out.sum().backward()
                equal([out.detach(),model.scale.grad,torch.get_rng_state()],list(original))
                expected=['publish_begin','first_fields_ready','forward_begin','forward_enqueued','publication_joined','fields_accepted','graph_consumed']
                require(list(early.stamps)==expected and list(early.stamps.values())==sorted(early.stamps.values()),
                        'overlap boundaries missing or out of order')
                try:early.consume({})
                except ValueError:pass
                else:raise ValueError('reused a consumed graph')
        shutil.rmtree(dst)  # keep only failed replay directories for diagnosis
    return dict(host_offsets_ms={k:round((v-early.stamps['publish_begin'])/1e6,3) for k,v in early.stamps.items()},cpu=True,rank=rank,real_publication=True,all_packet_fields_equal=True,dropout_output_gradient_rng_equal=True,
                first_graph_consumed_once=True)
if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--root',required=True,type=Path);args=cli.parse_args()
    sys.path.insert(1,'/sealed');print(json.dumps(replay(args.root)))
