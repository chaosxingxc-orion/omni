"""Minimal S-only adapter to existing native role/hooks, journal and generation lease.

check is CPU-only. run is for the existing NPU owner after actual reconciliation.
This owner never invokes run. Unknown journal states are never resent.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time
import threading
from builder import canonical,require,save,sha
from fit_s_inherited import validate_receipt


PROCESS_STARTED_EPOCH=time.time()
PROCESS_STARTED_MONOTONIC=time.monotonic()


WIRE_ADAPTER='core.model.vllm_transport.vllm_request_body'


def mapped_wire(packet, row):
    """Keep the immutable SDK packet; bind only the existing adapter's wire."""
    from speech_aware_evidence_acquisition.core.model.vllm_transport import vllm_request_body
    require(row.get('wire_adapter')==WIRE_ADAPTER,'corrected vLLM manifest required')
    abstract=packet['wire']
    require(sha(canonical(abstract))==row['abstract_wire_sha256'],'abstract wire changed')
    wire=vllm_request_body(abstract,abstract['model'])
    require(sha(canonical(wire))==row['wire_sha256'],'mapped wire changed')
    require(wire.get('thinking_token_budget')==3072 and 'thinking_budget_tokens' not in wire and
            wire['max_tokens']==4096 and wire['chat_template_kwargs']=={'enable_thinking':True},
            'mapped thinking/output policy')
    return wire


def native_completion(core, *, raw, request_id, sample_id, contact_id, request_sha256):
    """Preserve missing/null/[] exactly in the shared adapter envelope."""
    choice=json.loads(raw)['choices'][0];message=choice['message'];text=message['content']
    # SDK scorer text is a string, while original_message retains wire null.
    # The native bridge then raises its source-bound finite-completion error.
    if text is None and choice.get('finish_reason') in ('length','content_filter'):text=''
    require(isinstance(text,str),'no S answer content')
    return core.CoreCompletion(text=text,request_id=request_id,sample_id=sample_id,agent_contact_id=contact_id,
        request_sha256=request_sha256,response_sha256=sha(raw),response_text_sha256=sha(text.encode()),
        tool_calls_json=canonical(message.get('tool_calls')),wire_response_bytes=raw,
        tool_names=(),tool_schemas_sha256=sha(b'[]'))


class AdmissionStopped(Exception):
    """No intent was created; the contact remains absent."""


class Admission:
    """One local stop file; serialize admission against a durable stop ACK.

    Already admitted contacts must finish. Do not delete the file in a live run.
    The owner removes it only after PAUSED/exit/quiescence to resume explicitly.
    """
    def __init__(self, stop_file):
        self.stop_file=Path(stop_file)
        self.lock=threading.Lock();self.closed=False;self.reason=None
        self.admitted=0

    def _check(self):
        if self.stop_file.exists():
            self.closed=True;self.reason=self.reason or 'stop-file'
        return self.closed

    def stopped(self):
        with self.lock:return self._check()

    def stop(self, reason):
        with self.lock:
            self.closed=True;self.reason=self.reason or reason

    def admit(self, journal, custody):
        with self.lock:
            if self._check():raise AdmissionStopped('STOP_ADMISSION_NO_POST')
            journal.intent(custody);self.admitted+=1


def bind_executor(concurrency):
    require(type(concurrency) is int and concurrency>0,'positive concurrency required')
    executor=ThreadPoolExecutor(max_workers=concurrency,thread_name_prefix='s-http')
    asyncio.get_running_loop().set_default_executor(executor)
    return executor  # asyncio.run shuts down its default executor after drain.


async def observed_response(journal, custody, post, *, admission=None, timing=None):
    """Reuse raw once; admit in the actual worker, separately time pool wait/HTTP."""
    timing={} if timing is None else timing
    rid=custody['request_id'];state=journal.inspect(rid)
    if state['state']=='observed':
        require(state['intent']['payload']==custody,'replay custody differs')
        timing.update(executor_queue_seconds=0.0,admission_seconds=0.0,
                      new_http_seconds=0.0,source_latency_kind='legacy_unsplit_or_original_http')
        return state['response_bytes'],state['response']['latency_seconds'],True
    require(state['state']=='absent','UNKNOWN_NO_RESEND')
    submitted=time.monotonic()
    def deliver():
        started=time.monotonic()
        timing['executor_queue_seconds']=started-submitted
        if admission is None:journal.intent(custody)
        else:admission.admit(journal,custody)
        http_start=time.monotonic();timing['admission_seconds']=http_start-started
        raw=post();lat=time.monotonic()-http_start
        # Persist in this worker before returning to any SDK parsing/cancellation.
        journal.response(rid,sha(canonical(custody)),raw,lat)
        timing.update(new_http_seconds=lat,source_latency_kind='worker_post_seconds')
        return raw,lat,False
    return await asyncio.to_thread(deliver)


def observation_outcome(*,raw,flags,native_error,native,parser,candidate_ids):
    """Only typed finite model failures or completed invalid answers may continue."""
    require(flags.get('before') and flags.get('after'),'native hooks incomplete')
    obj=json.loads(raw);choice=obj['choices'][0];msg=choice['message']
    text=msg.get('content');finish=choice.get('finish_reason')
    if native_error is not None:
        finite=isinstance(native_error,native.FrozenCoreRoleFiniteCompletionError)
        require(finite and finish in ('length','content_filter') and
                native_error.finish_reason==finish and
                native_error.completion.wire_response_bytes==raw and
                native_error.completion.response_sha256==sha(raw),
                'program/native failure: '+type(native_error).__name__+': '+str(native_error))
        return dict(valid=False,outcome='known_model_invalid',model_failure=finish,
                    error=type(native_error).__name__+': '+str(native_error))
    require(finish=='stop','unsupported native finish reason')
    require(isinstance(text,str) and not msg.get('tool_calls'),'unexpected native response envelope')
    try:parser(text,candidate_ids)
    except (ValueError,KeyError,TypeError) as error:
        return dict(valid=False,outcome='known_model_invalid',model_failure='invalid_slot_output',error=str(error))
    return dict(valid=True,outcome='valid',model_failure=None,error=None)


async def execute_samples(packets,one,progress,results,*,concurrency=64,admission=None,can_replay=None):
    """Bound tasks, first/replay once; on errors stop admission and drain peers."""
    if not packets:return
    await one(*packets[0])
    if results:
        progress('FIRST_PROTOCOL_PASS' if results[-1]['valid'] else 'FIRST_KNOWN_MODEL_INVALID')
    remaining=iter(packets[1:])
    async def worker():
        while True:
            pair=next(remaining,None)
            if pair is None:return
            if admission is not None and admission.stopped():
                if can_replay is None:return
                if not can_replay(*pair):continue
            try:await one(*pair)
            except BaseException:
                if admission is not None:admission.stop('program-error')
                raise
    tasks=[asyncio.create_task(worker()) for _ in range(min(concurrency,len(packets)-1))]
    errors=await asyncio.gather(*tasks,return_exceptions=True)
    for error in errors:
        if isinstance(error,BaseException):raise error


def validate_native_message_bridge(*, request, packet, row, contact, flags, history):
    """Validate SDK representation; return the unchanged, hash-bound wire.

    NativeRoleNode strips the system prompt's outer whitespace. This permits
    that representation only; user text, roles/order and wire remain exact.
    Both live delivery and the offline SDK fixture use this entire boundary.
    """
    require(flags.get('before') is True and request.role_id=='slot' and not request.tool_schemas,
            'native hook/tools contract')
    require(request.context.sample_id==row['sample_id'] and request.context.agent_contact_id==contact,
            'native context')
    require(not history,'unexpected native history; refuse rather than discard')
    history.extend(copy.deepcopy(list(request.messages)))
    require(len(history)==2,'unexpected native history; refuse rather than discard')
    require([m.get('role') for m in history]==['system','user'],'native message roles/order differ')
    expected=packet['wire']['messages'][0]['content'];actual=history[0].get('content')
    require(isinstance(expected,str) and isinstance(actual,str) and
            (actual==expected or actual==expected.strip()),'native system prompt differs')
    require(history[1].get('content')==packet['wire']['messages'][1]['content'][-1]['text'],
            'native user text differs')
    wire=canonical(mapped_wire(packet,row))
    require(flags.get('wire_sha256')==sha(wire),'before/send mapped wire binding')
    bridge=dict(system_normalization=('exact' if actual==expected else 'sdk-strip'),
                expected_system_sha256=sha(expected.encode()),framework_system_sha256=sha(actual.encode()),
                expected_system_bytes=len(expected.encode()),framework_system_bytes=len(actual.encode()),
                framework_messages_sha256=sha(canonical(history)),wire_sha256=sha(wire))
    return wire,bridge


async def invoke_native_slot(*, native, ip, handoff_module, packet, row, snapshot,
                             snapshot_values, metered, contact, flags, response_state):
    """The shared real and fixture SDK path, including meaningful rails hooks."""
    from materialize_s import validate_request
    def before(role,ctx):
        require(role=='slot' and not flags.get('before'),'duplicate/wrong before hook')
        validate_request(packet,selected=row['candidate_ids'],expected_audio=packet['wire']['messages'][1]['content'][0],
            snapshot=snapshot,values=snapshot_values,handoff_module=handoff_module)
        require(packet['handoff']['source_response_sha256']==row['P_response_sha256'],'before P response binding')
        wire=mapped_wire(packet,row)
        flags['wire_sha256']=sha(canonical(wire))
        flags['response_format_sha256']=sha(canonical(wire['response_format']))
        flags['before']=True
    def after(role,ctx):
        require(role=='slot' and flags.get('before'),'after without before');flags['after']=True
        try:
            ip.slot_results(response_state['text'],packet['candidate_ids']);flags['slot_parse']='PASS'
        except (ValueError,KeyError,TypeError) as error:
            flags['slot_parse']='FAIL';flags['slot_error']=str(error)
    instruction=packet['wire']['messages'][0]['content']
    node=native.NativeRoleNode(spec=native.NativeRoleNodeSpec(role_id='slot',role_kind='slot',prompt=instruction,max_iterations=1),
        metered_request=metered,contact_context=native.NativeRoleContactContext(sample_id=row['sample_id'],agent_contact_id=contact,
          evidence_pins={'S_packet':row['packet_sha256'],'S_wire':row['wire_sha256'],
                         'P_response':row['P_response_sha256']}),candidate_before=before,candidate_after=after)
    try:
        return await node.invoke_direct(packet['wire']['messages'][1]['content'][-1]['text'])
    finally:
        await node.close()


def preflight(config, workspace):
    w=Path(workspace).resolve();c=json.loads(Path(config).read_bytes())
    require(c['cohort']=='fit','provisional baseline is fit only; freeze before validation')
    require(c.get('wire_adapter')==WIRE_ADAPTER,'corrected vLLM config required')
    path=w/c['manifest'];require(sha(path.read_bytes())==c['manifest_sha256'],'S manifest pin')
    rows=[json.loads(l) for l in path.open() if l.strip()];rows=[r for r in rows if r['cohort']=='fit']
    require(len(rows)==c['expected_rows'] and len({r['sample_id'] for r in rows})==c['expected_rows'],'exact configured fit roster')
    src=w/c['rails_src'];sys.path.insert(0,str(src))
    adapter=importlib.import_module('speech_aware_evidence_acquisition.core.model.vllm_transport')
    require(sha(Path(adapter.__file__).read_bytes())==c['wire_adapter_sha256'],'adapter source pin')
    hb=importlib.import_module('speech_aware_evidence_acquisition.rails.candidate_handoff')
    from materialize_s import validate_request
    snap=json.loads((w/c['snapshot']).read_bytes());values={v['value_id']:v['content'] for v in snap['values']}
    packets=[]
    for r in rows:
        p=w/r['payload'];raw=p.read_bytes();require(sha(raw)==r['packet_sha256'],'packet mutation')
        packet=json.loads(raw);require(packet['sample_id']==r['sample_id'],'sample mismatch')
        validate_request(packet,selected=r['candidate_ids'],expected_audio=packet['wire']['messages'][1]['content'][0],
             snapshot=c['snapshot_sha256'],values=values,handoff_module=hb)
        require(packet['handoff']['source_response_sha256']==r['P_response_sha256'],'P source mismatch')
        mapped_wire(packet,r)
        packets.append((r,packet))
    return c,packets


def load_execution(path, config, original_config):
    require(path is not None,'independent --execution-config required')
    spec=json.loads(Path(path).read_bytes())
    require(spec['schema']=='s-execution-efficiency-v1','execution schema')
    require(spec['original_runtime_config_sha256']==sha(Path(original_config).read_bytes()),'original runtime config pin')
    require(spec['run_id']==config['run_id'] and spec['attempt_id']==config['attempt_id'],'execution identity')
    require(spec['executor_max_workers']==config['concurrency'] and config['concurrency'] in (64,96,128) and (config['concurrency']==64 or config.get('dataset_partition') in ('S_fit','eligible_remainder')), 'explicit TRAIN concurrency64/96/128')
    require(spec['runner_sha256']==sha(Path(__file__).read_bytes()),'execution runner pin')
    require(spec['resources_module_sha256']==sha(Path(__file__).with_name('native_resources.py').read_bytes()),'resources module pin')
    return spec


async def run(config,workspace,output,execution_config=None):
    preparation_started=time.monotonic()
    c,packets=preflight(config,workspace);w=Path(workspace).resolve();out=Path(output).resolve()
    from native_resources import prepare_native_resources
    resources=prepare_native_resources()
    snapshot_doc=json.loads((w/c['snapshot']).read_bytes())
    snapshot_values={v['value_id']:v['content'] for v in snapshot_doc['values']}
    preparation_seconds=time.monotonic()-preparation_started
    execution=load_execution(execution_config,c,config)
    executor=bind_executor(c['concurrency'])
    admission=Admission(out/'STOP-ADMISSION')
    runtime=c['runtime'];require(runtime['status']=='OWNER_RECONCILED','runtime not reconciled')
    require(runtime['TP']==runtime['cards']==4 and runtime['instances']==1 and runtime['context_length']==65536,'single TP4 context binding')
    out.mkdir(parents=True,exist_ok=True)
    native=importlib.import_module('speech_aware_evidence_acquisition.reproduction.ojw.native_roles')
    core=importlib.import_module('speech_aware_evidence_acquisition.rails.core_client')
    ip=importlib.import_module('speech_aware_evidence_acquisition.rails.intent_slot_pipeline')
    cj=importlib.import_module('speech_aware_evidence_acquisition.core.contact_journal')
    cd=importlib.import_module('speech_aware_evidence_acquisition.core.contact_delivery')
    lease=importlib.import_module('speech_aware_evidence_acquisition.core.flight.lease')
    import urllib.request
    import urllib.error
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==runtime['boot_id'],'boot changed')
    require(os.stat(runtime['shared_guard']).st_ino==runtime['shared_guard_inode'],'shared guard changed')
    actual_guard,_ = lease._paths(Path(runtime['container_lane']))
    require(actual_guard.resolve()==Path(runtime['shared_guard']).resolve() and
            os.stat(actual_guard).st_ino==runtime['shared_guard_inode'],'lease uses a different shared guard')
    require(runtime['fixed_image_sha256']=='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026','image binding')
    for key in ('native_fit_receipt','predecessor_receipt'):
        p=Path(runtime[key]['path']);require(sha(p.read_bytes())==runtime[key]['sha256'],key+' pin')
    fit=json.loads(Path(runtime['native_fit_receipt']['path']).read_bytes())
    validate_receipt(fit,c,packets,w)
    pred=json.loads(Path(runtime['predecessor_receipt']['path']).read_bytes())
    require(pred['unknown']==0 and pred['status']=='COMPLETE','predecessor not COMPLETE')
    capture_path=Path(runtime['capture_receipt']['path'])
    require(sha(capture_path.read_bytes())==runtime['capture_receipt']['sha256'],'runtime capture pin')
    capture=json.loads(capture_path.read_bytes())
    require(capture['status']=='PASS_RUNTIME_CAPTURE_NOT_STARTED' and capture['released'] is True and capture['unknown']==0,'capture not admitted')
    for key in ('boot_id','session_id','service_pid','service_id'):
        require(capture[key]==runtime[key],'capture identity: '+key)
    require(capture['lane_root']==runtime['container_lane'] and capture['guard_inode']==runtime['shared_guard_inode'],'capture original lane')
    require(capture['predecessor']['receipt']==runtime['predecessor_receipt'],'capture predecessor')
    require(capture['fit_sha256']==runtime['native_fit_receipt']['sha256'],'capture fit')
    session=json.loads((Path(runtime['serving_root'])/runtime['session_id']/'session.json').read_bytes())
    require(session['session_id']==runtime['session_id'] and session['pid']==runtime['service_pid'],'session identity')
    require(session['identity']['port']==int(runtime['endpoint'].rsplit(':',1)[1]),'session endpoint')
    os.kill(runtime['service_pid'],0)
    localguard=(out/'generation.guard').open('a+b');fcntl.flock(localguard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    class Journal(cj.ContactJournal):
        def __init__(self):
            self.root=out/'contact-journals'/c['run_id']/c['attempt_id']
            self.identity=dict(run_id=c['run_id'],attempt_id=c['attempt_id'],arm='provisional-S-only-fit',
                  protocol_hash=sha(canonical({k:v for k,v in c.items() if k!='runtime'})),split_identity_hash=sha(canonical([r['sample_id'] for r,_ in packets])))
    journal=Journal();sem=asyncio.Semaphore(c['concurrency']);results=[];t=time.monotonic()
    observed_at_start=sum(journal.inspect('S-'+sha(canonical([c['run_id'],c['attempt_id'],r['sample_id'],r['wire_sha256']])))['state']=='observed' for r,_ in packets)
    original_launch_epoch=execution['original_launch_epoch']
    save(out/'resource-preparation'/(str(time.time_ns())+'.json'),dict(**resources,preparation_seconds=preparation_seconds,
        process_started_epoch=PROCESS_STARTED_EPOCH,observed_at_start=observed_at_start))
    def payload(r,p):
        contact='slot-'+sha(canonical([c['run_id'],r['sample_id']]))
        return dict(request_id='S-'+sha(canonical([c['run_id'],c['attempt_id'],r['sample_id'],r['wire_sha256']])),
          sample_id=r['sample_id'],history=[dict(role='control',contact_id=contact)],
          packet_sha256=r['packet_sha256'],wire_sha256=r['wire_sha256'],
          abstract_wire_sha256=r['abstract_wire_sha256'],wire_adapter=WIRE_ADAPTER,
          P_response_sha256=r['P_response_sha256'])
    expected_intents={journal._path(payload(r,p)['request_id'],'intent') for r,p in packets}
    require(set(journal.root.glob('*.intent.json'))<=expected_intents,'unexpected prior contact inventory')
    def progress(status,error=None):
        inventory=[journal.inspect(payload(r,p)['request_id']) for r,p in packets]
        states=[state['state'] for state in inventory]
        elapsed=time.monotonic()-t;observed=states.count('observed');unknown=sum(s not in ('absent','observed') for s in states)
        new_completed=observed-observed_at_start
        cumulative_elapsed=time.time()-original_launch_epoch
        usage={};historical_http=0.0
        for state in inventory:
            if state['state']!='observed':continue
            historical_http+=state['response']['latency_seconds']
            for k,v in json.loads(state['response_bytes']).get('usage',{}).items():
                if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
        doc=dict(status=status,completed=observed,rows_saved=len(results),total=len(packets),unknown=unknown,
            valid=sum(r['valid'] for r in results),known_model_invalid=sum(r.get('outcome')=='known_model_invalid' for r in results),
            reused_observations=sum(r.get('reused_observation',False) for r in results),
            logical_HTTP_seconds=historical_http,
            new_HTTP_seconds=sum(r['latency_seconds'] for r in results if not r.get('reused_observation',False)),
            seconds_per_completed=(cumulative_elapsed/observed if observed else None),wall_seconds=cumulative_elapsed,
            cumulative_elapsed_seconds=cumulative_elapsed,original_launch_epoch=original_launch_epoch,
            session_wall_seconds=elapsed,session_process_elapsed_seconds=time.monotonic()-PROCESS_STARTED_MONOTONIC,
            observed_at_session_start=observed_at_start,new_completed_this_session=new_completed,
            session_seconds_per_new_completed=(elapsed/new_completed if new_completed else None),
            preparation_seconds=preparation_seconds,
            remaining_seconds_estimate=(elapsed/new_completed*(len(packets)-observed) if new_completed else None),
            cards=4,instances=1,TP=4,concurrency=c['concurrency'],executor_max_workers=c['concurrency'],
            admission_closed=admission.stopped(),admission_reason=admission.reason,
            newly_admitted=admission.admitted,absent=states.count('absent'),
            executor_queue_seconds=sum(r.get('timing',{}).get('executor_queue_seconds',0) for r in results),
            measured_new_HTTP_seconds=sum(r.get('timing',{}).get('new_http_seconds',0) for r in results),
            legacy_latency_note='prior latency may include executor queue; never relabel as measured HTTP',
            usage=usage,money=None,
            new_P=0,S_logical_contacts=observed,final_P_frozen=False,error=error,at=time.time())
        save(out/'PROGRESS.json',doc)
        if status in ('FAILED','COMPLETE','PAUSED'):save(out/'TERMINAL.json',doc)
        return doc
    last_progress=[time.monotonic()]
    async def monitor():
        while True:
            await asyncio.sleep(1)
            if admission.stopped() and not (out/'DRAINING.json').exists():
                save(out/'DRAINING.json',dict(status='DRAINING',at=time.time(),reason=admission.reason,
                    admitted=admission.admitted,stop_file=str(admission.stop_file)))
                progress('DRAINING')
            elif time.monotonic()-last_progress[0]>=600:
                progress('DRAINING' if admission.stopped() else 'RUNNING');last_progress[0]=time.monotonic()
    async def one(r,p):
        async with sem:
            one_started=time.monotonic();sdk_timing={}
            custody=payload(r,p);rid=custody['request_id'];contact=custody['history'][0]['contact_id'];flags={'before':False,'after':False}
            framework_history=[];response_state={};message_bridge={}
            async def metered(request):
                sdk_timing['prepare_seconds']=time.monotonic()-one_started
                wire,bridge=validate_native_message_bridge(request=request,packet=p,row=r,contact=contact,
                    flags=flags,history=framework_history)
                message_bridge.update(bridge)
                def post():
                    require(sha(wire)==custody['wire_sha256']==flags['wire_sha256'],'journal/send wire binding')
                    with cd.delivery_scope(journal,rid,sha(canonical(custody))):
                        cd.begin_render(canonical(custody));cd.before_post(canonical(custody))
                        req=urllib.request.Request(runtime['endpoint']+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'})
                        try:
                            with urllib.request.urlopen(req,timeout=c['timeout_seconds']) as resp:return resp.read()
                        except urllib.error.HTTPError as error:
                            # Durable HTTP errors stay program/service failures.
                            return error.read()
                timing={}
                try:raw,lat,reused=await observed_response(journal,custody,post,admission=admission,timing=timing)
                except AdmissionStopped:
                    response_state['admission_stopped']=True
                    raise
                response_state['timing']=timing
                response_state['http_return_monotonic']=time.monotonic()
                obj=json.loads(raw);ch=obj['choices'][0];msg=ch['message'];text=msg.get('content')
                require(not msg.get('tool_calls'),'S must be tool-free')
                response_state.update(raw=raw,text=text,latency_seconds=lat,usage=obj.get('usage',{}),finish_reason=ch['finish_reason'],reused=reused)
                return native_completion(core,raw=raw,request_id=rid,sample_id=r['sample_id'],contact_id=contact,request_sha256=sha(wire))
            error=None;native_error=None;result=None
            try:
                result=await invoke_native_slot(native=native,ip=ip,handoff_module=importlib.import_module('speech_aware_evidence_acquisition.rails.candidate_handoff'),
                    packet=p,row=r,snapshot=c['snapshot_sha256'],snapshot_values=snapshot_values,
                    metered=metered,contact=contact,flags=flags,response_state=response_state)
            except Exception as e:native_error=e;error=type(e).__name__+': '+str(e)
            state=journal.inspect(rid)
            if response_state.get('admission_stopped'):
                require(state['state']=='absent','stopped contact must remain absent')
                return
            require(state['state']=='observed','native invocation failed before observed response: '+str(error))
            raw=state['response_bytes'];obj={};msg={};text='';finish=None
            try:
                obj=json.loads(raw);choice=obj['choices'][0];msg=choice['message'];text=msg.get('content','');finish=choice.get('finish_reason')
                require(isinstance(text,str),'no S answer content')
            except Exception as e:
                error=error or type(e).__name__+': '+str(e);text=''
                if not isinstance(obj,dict):obj={}
            outcome=dict(valid=False,outcome='program_error',model_failure=None,error=error)
            classification_error=None
            try:
                outcome=observation_outcome(raw=raw,flags=flags,native_error=native_error,native=native,
                    parser=ip.slot_results,candidate_ids=p['candidate_ids'])
            except Exception as e:
                classification_error=e;outcome['error']=type(e).__name__+': '+str(e)
            row=dict(cohort='fit',sample_id=r['sample_id'],content=text,usage=obj.get('usage',{}),
                latency_seconds=state['response']['latency_seconds'],**outcome,finish_reason=finish,
                raw_response_sha256=sha(raw),packet_sha256=r['packet_sha256'],hooks=flags,
                native_receipt=(result if result is not None else getattr(native_error,'execution_receipt',None)),
                reused_observation=response_state.get('reused',False),timing=response_state.get('timing',{}),wire_content=msg.get('content'),
                framework_messages=framework_history,message_bridge=message_bridge,
                wire_sha256=r['wire_sha256'],P_response_sha256=r['P_response_sha256'])
            row['timing'].update(sdk_timing)
            row['timing']['postprocess_seconds']=time.monotonic()-response_state.get('http_return_monotonic',time.monotonic())
            row['timing']['row_ready_epoch']=time.time()
            row_path=out/'rows'/(sha(r['sample_id'].encode())+'.private.json')
            # Preserve the earlier failed interpretation as well as journal raw.
            if row_path.exists() and row_path.read_bytes()!=canonical(row)+b'\n':
                save(out/'row-history'/row_path.stem/(str(time.time_ns())+'.json'),json.loads(row_path.read_bytes()))
            save(row_path,row);results.append(row)
            if classification_error is not None:raise classification_error
    monitor_task=None;execution_started=False
    try:
        for r,p in packets:require(journal.inspect(payload(r,p)['request_id'])['state'] in ('absent','observed'),'UNKNOWN_NO_RESEND')
        prior=[name for name in ('PROGRESS.json','TERMINAL.json','DRAINING.json') if (out/name).exists()]
        if prior:
            history=out/'status-history'/str(time.time_ns())
            for name in prior:save(history/name,json.loads((out/name).read_bytes()))
            if (out/'DRAINING.json').exists():(out/'DRAINING.json').unlink()
        if (out/'predictions.private.jsonl').exists():
            save(out/'prediction-history'/(str(time.time_ns())+'.jsonl'),
                [json.loads(l) for l in (out/'predictions.private.jsonl').read_bytes().splitlines()],lines=True)
        execution_started=True
        save(out/'runtime-bindings'/(str(time.time_ns())+'.json'),c)
        save(out/'implementation-history'/(str(time.time_ns())+'.json'),dict(runner_sha256=sha(Path(__file__).read_bytes()),
            outcome_policy='known-finite-model-invalid-continues-v1',run_id=c['run_id'],attempt_id=c['attempt_id'],
            original_runner_sha256=execution['original_runner_sha256'],
            execution_config_sha256=sha(Path(execution_config).read_bytes()),
            original_runtime_config_sha256=sha(Path(config).read_bytes()),
            protocol_hash=journal.identity['protocol_hash'],executor_max_workers=c['concurrency']))
        with lease.GenerationLease.acquire(Path(runtime['container_lane']),owner='s-correct-p-data-v1',run_id=c['run_id'],
               attempt_id=c['attempt_id'],session_id=runtime['session_id'],study_head=runtime['study_head']):
            # Recheck capture under the actual original GenerationLease.
            require(os.stat(actual_guard).st_ino==runtime['shared_guard_inode'],'guard replaced at admission')
            require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==runtime['boot_id'],'boot changed at admission')
            progress('RUNNING');monitor_task=asyncio.create_task(monitor())
            await execute_samples(packets,one,progress,results,concurrency=c['concurrency'],admission=admission,
                can_replay=lambda r,p:journal.inspect(payload(r,p)['request_id'])['state']=='observed')
            states=[journal.inspect(payload(r,p)['request_id'])['state'] for r,p in packets]
            require(all(s in ('absent','observed') for s in states),'DRAIN_UNKNOWN_NO_RESEND')
            require(states.count('observed')==len(results),'drain observed/rows differ')
            require(admission.stopped() or states.count('absent')==0,'incomplete cohort cannot COMPLETE')
            save(out/'predictions.private.jsonl',results,lines=True)
            progress('PAUSED' if admission.stopped() else 'COMPLETE')
    except BaseException as error:
        # Inflight to_thread delivery may still complete server-side. Preserve
        # GenerationLease metadata and UNKNOWN rather than dispatching again.
        if execution_started:
            save(out/'predictions.private.jsonl',results,lines=True)
            progress('FAILED',type(error).__name__+': '+str(error))
        else:
            save(out/'resume-rejections'/(str(time.time_ns())+'.json'),
                dict(status='RESUME_REFUSED',error=type(error).__name__+': '+str(error)))
        raise
    finally:
        if monitor_task:
            monitor_task.cancel();await asyncio.gather(monitor_task,return_exceptions=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);p.add_argument('--config',required=True);p.add_argument('--workspace',default='.');p.add_argument('--output');p.add_argument('--execution-config')
    a=p.parse_args()
    if a.mode=='check':
        c,packets=preflight(a.config,a.workspace);print(json.dumps(dict(status='CPU_INPUT_CHECK_PASS',fit=len(packets),native_sdk_executed=False,model_calls=0)))
    else:
        require(a.output is not None,'run requires local persistent output');asyncio.run(run(a.config,a.workspace,a.output,a.execution_config))
