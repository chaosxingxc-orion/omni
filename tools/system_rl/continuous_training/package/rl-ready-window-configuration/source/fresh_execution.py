"""Automatic native cost/version binding; execute only on explicitly supplied lease."""
import copy
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
from cohort_io import read,pin,save,require,storage,environment
from prompt_capacity import prompt_bound,request_cap


class CaptureAccounting:
    """Lease-local index; rebuild once on restart, commit only after durable save.

    The original exclusive owner is the only writer. This is not a replacement
    for Registry or HTTP reconciliation and never establishes request outcome.
    """
    def __init__(self, root, budget):
        self.root=Path(root);self.budget=budget;self.paths=set();self.tokens=0
        self.loaded_packets=0
        for path in self.root.glob('groups/*/episode-*/*.packet.json'):
            packet=json.loads(path.read_bytes());self.loaded_packets+=1
            self.record(path,packet)

    def charge(self, packet):
        maximum=packet['wire']['max_tokens']
        if 'sampled_request_limit' in self.budget:
            bound=self.budget['sampled_request_limit']
            require(packet['stage']==bound['node'] and type(maximum) is int and 0<maximum<=bound['max_tokens'],
                    'actual request exceeds sampled node or completion envelope')
        require(type(maximum) is int and maximum>0,'positive actual response cap')
        return request_cap(self.budget,packet)

    def check(self, path, packet):
        require(Path(path) not in self.paths,'existing packet must reconcile before contact')
        require(len(self.paths)<self.budget['max_requests'],'capture request ceiling')
        require(self.tokens+self.charge(packet)<=self.budget['capture_logical_tokens_upper'],
                'capture token ceiling')

    def record(self, path, packet):
        path=Path(path)
        require(path not in self.paths,'duplicate capture packet path')
        self.tokens+=self.charge(packet);self.paths.add(path)


class PromptLengths:
    """Bounded cache of pure processor results for identical wire messages.

    Retain only integer lengths, never waveforms, features, model outputs or
    tokens. A new lease/processor starts empty; audio bytes are part of the key.
    """
    def __init__(self, capacity=128):
        from collections import OrderedDict
        require(type(capacity) is int and capacity>0,'positive prompt cache capacity')
        self.capacity=capacity;self.values=OrderedDict();self.hits=0;self.misses=0

    def get(self, packet, prepare):
        from cohort_io import digest
        wire=packet['wire']
        key=digest({k:wire.get(k) for k in ('messages','chat_template_kwargs',
                                          'add_generation_prompt','continue_final_message')})
        if key in self.values:
            self.hits+=1;self.values.move_to_end(key);return self.values[key]
        value=prepare();require(type(value) is int and value>0,'actual processor prompt length')
        self.misses+=1;self.values[key]=value
        if len(self.values)>self.capacity:self.values.popitem(last=False)
        return value


def cost_plan(config,root,tokenizer):
    root=Path(root);native=copy.deepcopy(config['native']);budget=config['budget']
    # Use the ORIGINAL admitted PreparedBatch and configured task compiler.
    # Count physical padded token lengths, not merely generated action tokens.
    from cohort_io import HERE,BASE,ACL,imports,load
    consumer=load(HERE/'consumer.py','fresh_cost_consumer')
    fields,report=consumer.build(read(pin(root/'FILES.json')),read(pin(root/'CONSUMER.json')))
    events=[]
    if report['gradient_status']!='no_gradient':
        with imports([ACL,BASE/'native-training-successor']):
            from native_update import configure_imports,configured_plan
            configure_imports()
            bridge=load(BASE/'unified-autograd-bridge/bridge.py','fresh_cost_bridge')
            kernel=bridge.SourceKernel({n:pin(BASE/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
            prepared=bridge.PreparedBatch(fields,report,kernel)
            task_native=read(pin(root/'NATIVE.disabled.json'))
            _,plan,_=configured_plan(prepared,fields,root,task_native)
        width=fields['tensors']['input_ids'].shape[-1]
        prompt_width=width-fields['tensors']['responses'].shape[-1]
        events.extend(dict(kind='policy_forward_backward',row=i,tokens=width) for i in range(prepared.rows))
        for row,items in enumerate(plan.rows):
            for event in items:
                if plan.coefficients[event['term']]==0:continue
                targets=[event['preferred'],event['dispreferred']] if event['kind']=='ranking' else [event['text']]
                for target in targets:
                    text=target if isinstance(target,str) else target['text']
                    eos=1 if isinstance(target,str) or target.get('append_eos',True) else 0
                    size=prompt_width+len(tokenizer.encode(text,add_special_tokens=False))+eos
                    for kind in ('auxiliary_probe','auxiliary_recompute'):
                        events.append(dict(kind=kind,row=row,term=event['term'],tokens=size))
        # Original actor post-update and checkpoint reload each use first padded
        # row. Original publication performs R/C/P with the retained unpadded row.
        events.extend(dict(kind=k,row=0,tokens=width) for k in ('post_update','checkpoint_reload'))
        events.extend(dict(kind='publication_'+k,row=0,tokens=width+1) for k in ('R','C','P'))
    upper=sum(e['tokens'] for e in events)
    prior=budget['cumulative_prior_tokens_upper']
    capture_costs=read(pin(root/'CAPTURE-COSTS.json'))
    capture_upper=0
    for group in capture_costs:
        for call in group:
            usage=call['usage'];require(usage and type(usage['total_tokens']) is int,'actual capture usage required')
            capture_upper+=usage['total_tokens']+call['current_forward_tokens']+call['reference_forward_tokens']
    # The existing Registry is authoritative for historical/unknown costs and
    # capture reservations. This cohort's native upper is already reserved by
    # launch_native; replace only that reservation in the projected event plan.
    import sqlite3
    registry_path=Path(config['execution_registry']['path'])
    db=sqlite3.connect(registry_path.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        ncost=db.execute('SELECT COALESCE(SUM(cost),0) FROM spend').fetchone()[0]
        hcost=db.execute('SELECT COALESCE(SUM(cost),0) FROM history').fetchone()[0]
        rid='native-phase:'+read(pin(root/'SELECTED.json'))['version']
        own=db.execute('SELECT cost FROM spend WHERE id=?',(rid,)).fetchone()
        require(own and own[0]==budget['native_logical_tokens_upper'],'original native reservation missing')
        prior=int(ncost+hcost-own[0])
    finally:db.close()
    require(prior>=budget['cumulative_prior_tokens_upper']+capture_upper,'original cumulative ledger undercharges capture or inherited cost')
    require(upper<=budget['native_logical_tokens_upper'] and prior+upper<=budget['total_tokens'],'native cumulative token ceiling')
    ref=save(root/'NATIVE-COST-PLAN.json',dict(attempt_id=native['attempt_id'],logical_tokens_upper=upper,
        includes_recompute_and_auxiliary=True,includes_post_update_and_reload=True,kind='derived_original_event_plan_upper_bound',
        events=events,logical_DP_replication=4,DP_replication_multiplier=1,backward_and_activation_replay_in_currency_estimate=False,
        currency_cost=None,capture_logical_tokens=capture_upper,capture_costs_ref=pin(root/'CAPTURE-COSTS.json')))
    return dict(execution_cost_plan_ref=ref,limits=dict(total_tokens=budget['total_tokens'],this_attempt_tokens_upper=upper,
        cumulative_prior_tokens_upper=prior,cumulative_projected_upper=prior+upper))


def execution_modules():
    # Import original authority implementations in their own dependency scope.
    from cohort_io import BASE,CODE,imports,load
    owner=BASE/'collection-dispatch-integration'
    names=('base','bootstrap','fence','lease_runtime','common','transport','engine','hooks','reservations')
    with imports([owner/'generic-entry-successor',owner/'automatic-successor',owner,CODE],names):
        registry=load(owner/'reservations.py','fresh_original_registry')
        accounted=load(owner/'engine_adapter.py','fresh_original_accounted')
        fence=load(owner/'generic-entry-successor/fence.py','fresh_original_fence')
        original_probe = fence.probe

        def probe(remote):
            # The generic probe expects the config path from its execution host.
            # Preserve the local pin, and present its remote path only for SSH.
            remote_path = remote.get('config_path', remote['config']['path'])
            return original_probe(dict(remote, config=dict(remote['config'], path=remote_path)))

        fence.probe = probe
    return registry.Registry,accounted.AccountedTransport,fence.LeaseFence,fence.ContactFence


class CaptureLease:
    """Borrow EXISTING original Registry + remote LeaseFence; never acquire/release.

    The resource owner already owns REMOTE-LEASE and Registry. Unknown spend
    cannot be cleared by this bridge. Original fence safety probes may use SSH
    only in a later explicitly enabled execution, never CPU configuration.
    """
    def __init__(self,config,root):
        self.config=config;self.root=Path(root);self.processor=None;self.fence=None
        self.accounting=None;self.prompt_lengths=PromptLengths()
    def __enter__(self):
        c=self.config;auth=c['authorization']
        require(auth['model_calls'] is True and dt.datetime.fromisoformat(auth['expires_at'])>dt.datetime.now(dt.timezone.utc),'fresh execution authority missing/expired')
        Registry,self.Accounted,Fence,self.ContactFence=execution_modules()
        r=c['execution_registry'];path=Path(r['path']);storage(path.parent)
        require(path.is_file() and r['cost_unit']=='logical_tokens_upper','existing original token Registry required')
        self.registry=Registry(path,read(r['authority_ref']),read(r['resources_ref']))
        require(c['resources']['rank'] in self.registry.resources,'original resource rank missing')
        require(self.registry.authority['max_cost_units']<=c['budget']['total_tokens'] and self.registry.usage()['cost_units']>=c['budget']['cumulative_prior_tokens_upper'],'original cumulative budget cannot reset')
        require(not self.registry.admission_closed(),'original global admission closed')
        if 'session' in c['lease_fence']:
            from session_lease import LocalLease
            self.fence=LocalLease(c['lease_fence'])
        else:self.fence=Fence(c['lease_fence'])
        self.fence.start()
        with (self.root/'capture-lease-publish.guard').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            save(self.root/'CAPTURE-LEASE.json',dict(lease_fence=c['lease_fence'],registry=r,authorization=auth,automatic_next_cohort=False))
        return self
    def transport(self,inner):
        lease=self
        class Bound:
            def __getattr__(self,key):return getattr(inner,key)
            def submit(self,rid,packet):
                # Charge actual HTTP upper plus current/frozen-reference capture
                # BEFORE provider contact, through the SAME original ledger.
                cap=request_cap(lease.config['budget'],packet)
                # A failed pre-contact lease check must not create an unknown
                # spend. The accounted transport still reserves before HTTP,
                # and there is no retry after entry into that transport.
                accounted=lease.Accounted(inner,lease.registry,lease.config['resources']['rank'],3*cap)
                return lease.ContactFence(accounted,lease.fence).submit(rid,packet)
            def reconcile(self,rid,packet):
                return lease.Accounted(inner,lease.registry,lease.config['resources']['rank'],0).reconcile(rid,packet)
            def inspect(self,packet):
                state=inner.inspect(packet)
                with lease.registry.transaction() as db:
                    row=db.execute('SELECT status FROM spend WHERE id=?',(packet['request_id'],)).fetchone()
                # Crash between global reservation and native intent is unknown.
                if row and state['state']=='absent':return dict(state='global_intent_unknown')
                return state
        return Bound()
    def check(self,packet,transport):
        c=self.config;storage(self.root,c['capture']['runtime_storage_budget_bytes'])
        from boundary_acquisition import check_packet
        check_packet(c,packet)
        self.fence.check_local()
        require(not self.registry.admission_closed(),'original cumulative authority closed')
        require(dt.datetime.fromisoformat(c['authorization']['expires_at'])>dt.datetime.now(dt.timezone.utc),'fresh request deadline')
        if self.processor is None:
            from transformers import Qwen3OmniMoeProcessor
            self.processor=Qwen3OmniMoeProcessor.from_pretrained(c['native']['model_path'],local_files_only=True)
        from cohort_capture import prepare_packet
        prompt_tokens=self.prompt_lengths.get(packet,lambda:
            int(prepare_packet(self.processor,packet)['encoded']['input_ids'].shape[-1]))
        bound=prompt_bound(c['budget'],packet.get('stage'))
        require(prompt_tokens<=bound,'actual multimodal prompt exceeds configured bound: '+
                str(dict(node=packet.get('stage'),actual=prompt_tokens,bound=bound)))
        context=c.get('prompt_context_limit')
        if context is not None:
            require(prompt_tokens+packet['wire']['max_tokens']<=context,'actual prompt/completion exceeds consumer context')
        if self.accounting is None:self.accounting=CaptureAccounting(self.root,c['budget'])
        packet_path=transport.root/(packet['request_id']+'.packet.json')
        prepared=packet_path in self.accounting.paths
        if prepared:
            # A bounded wave may stop after preparing a packet but before its
            # registry intent. The caller can enter here only when both ledgers
            # report absence. Reuse the exact prepared packet in that case.
            require(read(pin(packet_path))==packet,'prepared request packet drift')
        else:
            self.accounting.check(packet_path,packet)
        usage=self.registry.usage()
        cap=request_cap(c['budget'],packet)
        require(usage['cost_units']+3*cap+c['budget']['native_logical_tokens_upper']<=c['budget']['total_tokens'],'capture/current/reference/native cumulative ceiling')
        if not prepared:
            save(packet_path,packet)
            self.accounting.record(packet_path,packet)
    def reserve_native(self):
        c=self.config;rid='native-phase:'+read(pin(self.root/'SELECTED.json'))['version']
        packet=dict(kind='original_native_update_and_publication',input_ref=pin(self.root/'INPUT.json'))
        with self.registry.transaction() as db:old=db.execute('SELECT status FROM spend WHERE id=?',(rid,)).fetchone()
        require(old is None,'existing native phase intent must reconcile; never repeat optimizer')
        with self.fence.admission():
            self.registry.reserve_request(rid,packet,c['resources']['rank'],c['budget']['native_logical_tokens_upper'])
        return rid
    def __exit__(self,*_):
        if self.fence:self.fence.close()


def arm(config,root,*,enabled=False):
    """Generate original native task layout; no device/controller creation."""
    root=Path(root);require((root/'GENERATION-CLOSED.json').exists(),'selected generation not closed')
    task=Path(config['native_task_root']).resolve();environment(task);(task/'successor').mkdir(exist_ok=True);(task/'hot').mkdir(exist_ok=True)
    native=read(pin(root/'NATIVE.pending.disabled.json'));native.update(enabled=enabled)
    require(native['learning_config_ref']==config['learning_config_ref'],'native learning binding')
    native['fresh_execution_input_ref']=pin(root/'INPUT.json')
    ref=save(task/'successor'/('config.real.json' if enabled else 'config.plan.disabled.json'),native)
    result=dict(schema='configured-native-launch-plan-v1',config_ref=ref,training_ready=False,
        command=['python3','-B',str(Path(__file__).with_name('launch_native.py')),'--config',str(root/'INPUT.json'),'--enable-native-training'],
        environment=dict(NATIVE_TASK_ROOT=str(task)),execution_enabled=enabled,
        lifecycle='existing resource owner starts fixed image; original ranks capture -> configured_update -> original checkpoint -> original sync publication',
        automatic_next_cohort=False)
    save(root/('NATIVE-EXECUTION.json' if enabled else 'NATIVE-PLAN.json'),result)
    return result


def verify_native_lease(native):
    config=read(native['fresh_execution_input_ref'])
    _,_,Fence,_=execution_modules()
    fence=Fence(config['lease_fence'])
    fence.validate()
    require(config['authorization']['native_training'] is True,'native execution authority missing')
    require(dt.datetime.fromisoformat(config['authorization']['expires_at'])>dt.datetime.now(dt.timezone.utc),'native execution authority expired')
    storage(Path(config['native_task_root'])/'hot',native['runtime_storage_budget_bytes'])
