"""Overlap the first policy forward with CPU publication of one immutable cohort."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dependencies import require
from microbatch_fields import CommitGate
from microbatch_producer import Producer
from microbatch_roster import group_rewards


class FirstWave:
    def __init__(self,spec,runtime,rank,world,worker,open_geometry):
        self.spec=spec
        self.runtime=runtime
        self.rank=rank
        self.world=world
        self.worker=worker
        self.open_geometry=open_geometry
        self.scores=None
        self.validated=False
        self.stamps={}

    def __call__(self,folder,fields,report):
        from pack_actor_fields import seal_packet
        seal_packet(folder,fields,report,self.spec['generation']['generation_inventory_ref'])

    def first_inputs(self,directory,cfg,plan,clock,learning,origins,labels_for):
        from cohort_io import read,pin
        from cohort import group_storage
        from fresh_adapter import consumer
        from dependencies import load
        snap=clock['groups'][plan['group_ids'][0]]['snapshot']
        require(all(clock['groups'][g]['snapshot']==snap for g in plan['group_ids'])
                and snap['target_nodes']==['P'] and not any(snap['auxiliary_coefficients'].values()),'unexpected policy scope')
        source=Path(self.runtime['consumer_source'])
        relative=self.spec['generation']['runtime']['sources']['consumer_source']
        expected=self.spec['capture_mounts']['overlays'][relative]
        def member(draw,name):
            return read(pin(directory/'groups'/str(draw['draw_index'])/name))
        wrapper=consumer(load(self.runtime['consumer_wrapper'],'first_wave_consumer'))
        with wrapper.unified(cfg) as (learner,_,kernel):
            env=learner.evaluate.__globals__
            def rewards(draw,group):
                return group_rewards(draw,group,snap,learning,labels_for=labels_for,score_episode=env['score_episode'],
                    reward=learner.reward,advantages=env['reduce_validated'].__globals__['advantages'])
            geometry=self.spec['actor_batch']
            produced=Producer(plan,(member(d,'GENERATED.json') for d in plan['draws']),snap,
                    world=self.world,micro=geometry['microbatch_per_card'],accumulation=geometry['gradient_accumulation_steps'],
                    group_size=learning['kernel']['G'],source=source,expected=expected,pad=cfg['pad_token_id'],
                    context_limit=cfg['context_limit'],reward_rows=rewards,origins=origins,
                    group_reader=lambda draw:group_storage(member(draw,'GROUP.json'),expand=True))
        if not produced.keys:return None
        self.slots=produced.layout['waves'][0][self.rank]
        _,t,n=produced.wave([self.slots])[0]
        self.gate=CommitGate([self.slots],dict(snapshot=snap,
                    global_episode_denominator=produced.layout['global_episode_denominator']))
        self.gate.observe(self.slots,t,n)
        return t,n

    def overlap(self,directory,config,consumer_config,plan,clock,learning,origins,labels_for,cost,consumer_ref,encoding):
        import torch
        from resident import collectively
        from bounded_actor_step import compact_scores
        from run_package import write
        directory=Path(directory)
        self.stamps['publish_begin']=time.monotonic_ns()
        def start():
            if self.rank:return None
            require(cost is not None,'missing actual reserved cost')
            args=dict(directory=str(directory),config=config,cost=cost,consumer_ref=consumer_ref,encoding=encoding,
                      spec=self.spec)
            ref=write(directory/'CPU-PUBLISH-INPUT.json',args)
            # Start a clean interpreter. The publisher's temporary imports and
            # mutable module registries never share an interpreter with FSDP.
            log=(directory/'CPU-PUBLISH.log').open('ab')
            try:
                return subprocess.Popen([sys.executable,'-B',__file__,'--publish',json.dumps(ref)],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,
                        PYTHONDONTWRITEBYTECODE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0'))
            finally:log.close()
        proc=collectively(start)
        try:
            fields=collectively(lambda:self.first_inputs(directory,consumer_config,plan,clock,learning,origins,labels_for))
            self.stamps['first_fields_ready']=time.monotonic_ns()
            if fields is not None:
                self.open_geometry()
                if self.spec.get('deterministic_backward'):torch.use_deterministic_algorithms(True,warn_only=False)
                self.worker.actor.actor_module.train()
                t,n=fields;device=next(self.worker.actor.actor_module.parameters()).device
                def convert():
                    result={**{k:v.to(device) for k,v in t.items()},**n}
                    result['multi_modal_inputs']=[{k:v.to(device) for k,v in mm.items()} for mm in n['multi_modal_inputs']]
                    return result
                inputs=collectively(convert)
                self.stamps['forward_begin']=time.monotonic_ns()
                self.scores=compact_scores(self.worker.actor,inputs)
                self.stamps['forward_enqueued']=time.monotonic_ns()
                require(self.scores.requires_grad and self.scores.grad_fn is not None,'missing policy graph')
            def finish():
                if proc is not None:
                    require(proc.wait(timeout=self.spec['timeout_seconds'])==0,'CPU publication failed; see CPU-PUBLISH.log')
            collectively(finish)
            self.stamps['publication_joined']=time.monotonic_ns()
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill();proc.wait()

    def bind(self,packet,fields,plan):
        from resident import collectively
        from events import partition
        def accept():
            if self.scores is None:
                require(plan.base.rows==0,'first graph missing for nonempty batch')
                return
            geometry=self.spec['actor_batch']
            micro=geometry['microbatch_per_card']
            actual=partition(plan,fields,self.world,micro,geometry['gradient_accumulation_steps'])['schedules'][self.rank][:micro]
            slots=[(e['row'],e['owned']) for e in actual]
            require(all(e['kind']=='policy' for e in actual),'unexpected auxiliary row')
            self.gate.reconcile(packet,[slots])
            self.identities=[plan.base.identities[i] for i,_ in slots]
        collectively(accept)
        self.validated=True
        self.stamps['fields_accepted']=time.monotonic_ns()

    def consume(self,inputs):
        require(self.validated and self.scores is not None and list(inputs['record_identity'])==self.identities,
                'first graph not bound to the actual microbatch')
        self.stamps['graph_consumed']=time.monotonic_ns()
        scores,self.scores=self.scores,None
        return scores


def publish(request):
    from run_package import read
    from bounded_actor_step import generation_sources
    from dependencies import load
    args=read(request);spec=args['spec']
    with generation_sources(spec) as rt:
        from fresh_adapter import scope
        from pack_actor_fields import seal_packet
        with scope():
            from cohort_io import save,read as captured_read
            cohort=load(rt['cohort_source'],'cohort')
            root=Path(args['directory'])
            outcome=cohort.publish(args['config'],root,consumer_ref=args['consumer_ref'],row_encoding=args['encoding'],
                accepted_sink=lambda folder,fields,report:seal_packet(folder,fields,report,spec['generation']['generation_inventory_ref']))
            native=captured_read(outcome['native_config_ref'])
            native.update(args['cost'],enabled=True)
            native.pop('pending_capture_ref',None)
            save(root/'NATIVE.accepted.json',native)


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--publish':raise ValueError('expected sealed CPU publisher input')
    sys.path.insert(1,'/sealed')
    publish(json.loads(sys.argv[2]))
