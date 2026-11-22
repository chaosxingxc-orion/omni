"""Mount source-matched timestamp probes in one diagnostic copy of a measured lane."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from project_prefill_tile import build as tile_parent
from project_screen_lanes import save,read
from decode_lineage import snapshot,digest

HELPER = '''
import json, os, time
now=time.monotonic_ns
_fd=None
def emit(kind, **fields):
    global _fd
    if _fd is None:
        _fd=os.open('/runtime/state/stages-'+str(os.getpid())+'.jsonl',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    os.write(_fd,(json.dumps(dict(t=now(),kind=kind,**fields),separators=(',',':'))+'\\n').encode())
'''

def inject(source, function, prefix, before='', after='', count=1):
    """Insert around complete statements; refuse changes to the expected function ABI."""
    module=ast.parse(source); lines=source.splitlines(keepends=True)
    matches=[]
    for f in ast.walk(module):
        if not isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)) or f.name!=function:continue
        for n in ast.walk(f):
            if isinstance(n,ast.stmt) and lines[n.lineno-1].lstrip().startswith(prefix):
                # Only match the statement at its own indentation, not enclosing expressions.
                if n.col_offset==len(lines[n.lineno-1])-len(lines[n.lineno-1].lstrip()):
                    matches.append((n.lineno-1,n.end_lineno,n.col_offset))
    if len(matches)!=count:raise ValueError((function,prefix,matches))
    for a,b,indent in sorted(matches,reverse=True):
        def fmt(s):return [' '*indent+v+'\n' for v in s.splitlines()]
        lines[b:b]=fmt(after);lines[a:a]=fmt(before)
    return ''.join(lines)


def build(spec):
    root=Path(spec['projection']['target']); parent=Path(spec['measured_parent'])
    tile_parent(spec); bind(spec,root,parent)

def bind(spec,root,parent):
    if snapshot(root)!=snapshot(parent):raise ValueError('diagnostic parent differs')
    lane,=spec['projection']['lanes']; code=root/('lane-'+lane); cfg=read(code/'config.json')
    source=spec['source_container']
    image=subprocess.check_output(['docker','inspect','-f','{{.Image}}',source],text=True).strip()
    if image!=cfg['image']:raise ValueError('source image differs')
    base='/vllm-workspace/vllm/vllm/'
    operator='/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'
    mapping={}; hashes={}
    dest=code/'probes';dest.mkdir()
    def add(name,target,data):
        path=dest/name;compile(data,str(path),'exec');path.write_text(data)
        mapping['probes/'+name]=target
    add('stage_clock.py',base+'stage_clock.py',HELPER)
    def original(target):
        p=subprocess.check_output(['docker','exec',source,'cat',target],text=True)
        import hashlib
        hashes[target]=hashlib.sha256(p.encode()).hexdigest()
        # Import lazily in instrumented statements so future imports and
        # module docstrings retain their original placement.
        return p
    def patch(target, edits):
        s=original(target)
        for fn,anchor,b,a in edits:
            s=inject(s,fn,anchor,b,a)
        add(Path(target).name,target,s)
    timer='from vllm import stage_clock as sc'
    patch(base+'entrypoints/openai/chat_completion/serving.py',[
        ('_create_chat_completion','result = await self.render_chat_request',timer+'\n_rt=sc.now()', ''),
        ('_create_chat_completion','request_metadata = RequestResponseMetadata', "sc.emit('render',rid=request_id,start=_rt,seed=request.seed)",'')])
    patch(base+'v1/engine/async_llm.py',[
        ('add_request','self.input_processor.assign_request_id(request)',timer,"sc.emit('map',rid=request_id,internal=request.request_id)"),
        ('_add_request','await self.engine_core.add_request_async(request)',timer+"\nsc.emit('ipc_send',rid=request.request_id)",'')])
    patch(base+'v1/core/sched/scheduler.py',[
        ('add_request','existing = self.requests.get(request.request_id)',timer+"\nsc.emit('enqueue',rid=request.request_id,prompt=request.num_prompt_tokens)",'')])
    edits=[]
    for fn in ('step','step_with_batch_queue'):
        edits+=[(fn,'scheduler_output = self.scheduler.schedule',timer+'\n_st=sc.now()',
                 "_sn=sc.now()\nsc.emit('scheduled',start=_st,end=_sn,key=id(scheduler_output),alloc=scheduler_output.num_scheduled_tokens,gen={k:v.num_output_tokens for k,v in self.scheduler.requests.items()},enc=list(scheduler_output.scheduled_encoder_inputs),waiting=len(self.scheduler.waiting),budget=self.scheduler.max_num_scheduled_tokens,seq_limit=self.scheduler.max_num_running_reqs,encoder_budget=self.scheduler.max_num_encoder_input_tokens)"),
                (fn,'model_output = future.result()',timer+'\n_wait=sc.now()', "sc.emit('wait',start=_wait,key=id(scheduler_output))"),
                (fn,'engine_core_outputs = self.scheduler.update_from_output','_update=sc.now()',
                 "sc.emit('update',start=_update,key=id(scheduler_output),out={o.request_id:len(o.new_token_ids) for group in engine_core_outputs.values() for o in group.outputs})")]
    edits += [('step','future = self.model_executor.execute_model','_submit=sc.now()', "sc.emit('submit',start=_submit,key=id(scheduler_output))"),
              ('step_with_batch_queue','exec_future = self.model_executor.execute_model','_submit=sc.now()', "sc.emit('submit',start=_submit,key=id(scheduler_output))"),
              ('step','model_output = self.model_executor.sample_tokens','_sample=sc.now()', "sc.emit('sample',start=_sample,key=id(scheduler_output))")]
    patch(base+'v1/engine/core.py',edits)
    patch(operator,[
        ('execute_model','self._cpp_execution_time_ms = None',timer+"\n_tbegin=sc.now()", ''),
        ('execute_model','hidden_states = self._model_forward','_tfwd=sc.now()',
         "sc.emit('worker_forward',start=_tfwd,entry=_tbegin,alloc=scheduler_output.num_scheduled_tokens)")])
    def once(s,a,b):
        if s.count(a)!=1:raise ValueError('unexpected middleware or launch source: '+a)
        return s.replace(a,b)
    b=root/'bundle/parents/jev-system-rl/implementation/service_bootstrap.py'
    s=b.read_text();s=once(s,"            generation=scope.get('path')=='/v1/chat/completions'", "            "+timer+"\n            scope['_trace_entry']=sc.now()\n            generation=scope.get('path')=='/v1/chat/completions'")
    b.write_text(s)
    b=b.with_name('service_overlay.py');s=b.read_text()
    s=once(s,'        parts = []','        '+timer+'\n        _body_start=sc.now()\n        parts = []')
    s=once(s,'        call_ref = self.calls.get(wire_sha)',"        sc.emit('http',wire=wire_sha,start=scope.get('_trace_entry',_body_start),body_start=_body_start)\n        call_ref = self.calls.get(wire_sha)")
    s=once(s,'                await send(message)',"                await send(message)\n                if message['type']=='http.response.body' and not message.get('more_body',False):sc.emit('sent',wire=wire_sha)")
    b.write_text(s)
    p=code/'launch.py';s=p.read_text();anchor='    for src, dst, ro in mounts:'
    p.write_text(once(s,anchor,'    mounts.extend((str(CODE/k),v,True) for k,v in '+repr(mapping)+'.items())\n'+anchor))
    pin=read(code/'PINS.json');pin.update({k:digest(code/k) for k in ['launch.py',*mapping]});save(code/'PINS.json',pin)
    save(root/'TRACE-ABI.json',dict(image=image,sources=hashes,patched=pin,parent=digest(parent/'CANDIDATE.json')))
if __name__=='__main__':build(read(Path(sys.argv[1])))
