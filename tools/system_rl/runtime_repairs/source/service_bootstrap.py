"""Fixed vLLM API-server bootstrap installs the overlay in its actual process."""
import argparse
import asyncio
from pathlib import Path
import sys
import os
import inspect
from io_utils import BASE,HERE,doc,pin,require

def worker_process_environment():
    """Mirror official CLI spawn setup before importing the vLLM runtime.

    This main-guarded server may initialize the accelerator before creating
    workers. Fork cannot reinitialize NPU state; reject an explicit conflict.
    """
    key='VLLM_WORKER_MULTIPROC_METHOD'
    require(os.environ.get(key,'spawn')=='spawn','NPU_HTTP_WORKERS_REQUIRE_SPAWN')
    os.environ[key]='spawn'

def custom_operator_environment():
    """Register vendor discovery before API imports can initialize CANN.

    Spawned workers inherit the vendor path before their device-type query.
    Keep the original fused operator; never disable it to hide load failures.
    """
    import torch_npu
    require(not torch_npu.npu.is_initialized(),'CUSTOM_OP_ENV_MUST_PRECEDE_NPU_INIT')
    from vllm_ascend.utils import bootstrap_custom_op_env
    bootstrap_custom_op_env()
    paths=[Path(p) for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if p]
    require(paths and all(p.is_dir() for p in paths),'CUSTOM_OP_VENDOR_PATH_MISSING')
    # is_dir alone does not prove that the worker UID can traverse a vendor
    # package or dlopen its root:root 750 libraries. Fail before model loading.
    libraries=[p/'op_api/lib/libcust_opapi.so' for p in paths]
    require(any(os.access(p,os.R_OK) and os.access(p.parent,os.X_OK) for p in libraries),
            'CUSTOM_OP_VENDOR_LIBRARY_UNREADABLE_FOR_SERVICE_UID')
    require(not torch_npu.npu.is_initialized(),'CUSTOM_OP_BOOTSTRAP_INITIALIZED_NPU')

def api_binding():
    """Bind hooks at the module that actually executes build_and_serve.

    New vLLM keeps api_server as a deprecated re-export. Patching that facade
    would silently leave the real server without the policy/precision overlay.
    """
    from importlib import import_module,util
    if util.find_spec('vllm.entrypoints.launchers') is not None:
        entry=import_module('vllm.entrypoints.launchers.api_server.entry')
        cli=import_module('vllm.entrypoints.launchers.cli_args')
    else:
        entry=import_module('vllm.entrypoints.openai.api_server')
        cli=import_module('vllm.entrypoints.openai.cli_args')
    require('args' in inspect.signature(entry.run_server).parameters,'PINNED_API_RUN_SERVER_SIGNATURE')
    require(callable(cli.make_arg_parser) and callable(cli.validate_parsed_serve_args) and
            callable(entry.build_app) and callable(entry.init_app_state),'PINNED_API_BOOTSTRAP')
    return entry,cli

def parse_service_args(cli,argv):
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser=cli.make_arg_parser(FlexibleArgumentParser(description='Original vLLM HTTP with fixed policy overlay'))
    # The pinned CLI restricts strings to half types, while the already-used
    # Python engine API accepts torch.float32 for the scoped Ascend LoRA repair.
    # Expose that exact explicit choice; retain every other parser constraint.
    action=next(a for a in parser._actions if '--lora-dtype' in a.option_strings)
    require(action.choices is not None,'explicit LoRA CLI dtype choices required')
    action.choices=list(dict.fromkeys([*action.choices,'float32']))
    argv=list(argv)
    require('--no-enable-prefix-caching' not in argv,'RL_ROLLOUT_PREFIX_CACHE_REQUIRED')
    if '--enable-prefix-caching' not in argv:argv.append('--enable-prefix-caching')
    parsed=parser.parse_args(argv)
    cli.validate_parsed_serve_args(parsed)
    require(parsed.tensor_parallel_size==4 and parsed.enable_lora and parsed.max_loras==1 and
        str(parsed.lora_dtype)=='float32' and parsed.worker_extension_cls=='ascend_fp32_worker.FP32LoRAWorkerExtension' and
        parsed.generation_config=='vllm' and parsed.logprobs_mode=='raw_logprobs','ACTUAL_API_MODEL_ADAPTER_DISTRIBUTION_ARGS')
    require(getattr(parsed,'api_server_count',1) in (None,1),'ONE_API_PROCESS_PER_OVERLAY')
    require(parsed.enable_prefix_caching is True,'RL_ROLLOUT_PREFIX_CACHE_REQUIRED')
    require(parsed.served_model_name==['qwen3-omni-30b-a3b-instruct'] or
            parsed.served_model_name=='qwen3-omni-30b-a3b-instruct','BUSINESS_SERVICE_NAME')
    return parsed


def build_with_overlay(original,middleware,*args,**kwargs):
    """Use vLLM's middleware registration before its eager stack bootstrap.

    Recent build_app seals the Starlette stack (SageMaker bootstrap) before
    returning. Register through the existing args.middleware extension point;
    never reset or replace that initialized stack.
    """
    import types
    bound=inspect.signature(original).bind(*args,**kwargs)
    parsed=bound.arguments['args']
    previous=parsed.middleware
    require(type(previous) is list,'explicit API middleware list required')
    name='omni_policy_runtime_middleware'
    require(name not in sys.modules,'policy middleware namespace already owned')
    module=types.ModuleType(name);module.PolicyMiddleware=middleware
    sys.modules[name]=module
    parsed.middleware=[*previous,name+'.PolicyMiddleware']
    try:return original(*args,**kwargs)
    finally:
        parsed.middleware=previous
        require(sys.modules.get(name) is module,'policy middleware namespace changed')
        del sys.modules[name]


def main():
    worker_process_environment()
    split=sys.argv.index('--')
    check_only='--check-only' in sys.argv[1:split]
    if not check_only:
        custom_operator_environment()
    config=doc(pin(sys.argv[sys.argv.index('--config')+1]))
    argv=sys.argv[split+1:]
    from sync_overlay import verify
    overlay_root=verify(config['named_rpc_overlay'])
    acl=BASE/'native-recompute-integration/state-repair/acl-runtime-successor'
    implementation=BASE/'jev-system-rl/implementation'
    sys.path[:0]=[overlay_root,str(acl),str(implementation)]
    os.environ['PYTHONPATH']=':'.join([overlay_root,str(acl),str(implementation),os.environ.get('PYTHONPATH','')])
    local=config.get('local_activation')
    if local:
        from local_activation import prepare, owner_check, install_precision
        prepare(local, overlay_root)
        if check_only:
            # Ascend API CLI imports Triton, which queries the SoC. The
            # device-free check validates the closed inputs; the real server
            # validates its argument parser before building an engine.
            from family_precision import validate
            from publication import validate_publication
            import importlib.util
            require(importlib.util.find_spec('acl') is not None,'CANN Python bindings missing from image environment')
            validate(doc(config['named_rpc_overlay']['precision_ref']))
            policy=doc(config['policy_ref'])
            validate_publication(policy['publication_ref'])
            doc(config['profile_ref'])
            return
        if local.get('diagnostic_logits'):
            import service_overlay
            from rollout_logits_service import overlay_class
            service_overlay.ServiceOverlay=overlay_class(service_overlay.ServiceOverlay,
                service_overlay.CURRENT,local['diagnostic_logits'])
    api_server,cli=api_binding()
    from service_overlay import ServiceOverlay
    original_init=api_server.init_app_state
    original_build=api_server.build_app
    handles={}

    async def init(*args,**kwargs):
        import inspect
        result=original_init(*args,**kwargs)
        if inspect.isawaitable(result):await result
        values=list(args)+list(kwargs.values())
        state=next(v for v in values if getattr(v,'openai_serving_chat',None) is not None)
        chat=state.openai_serving_chat
        handles.update(chat=chat,models=state.openai_serving_models,engine=chat.engine_client)
        stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        config['service_identity']=dict(config['service_identity'],pid=os.getpid(),
            start_ticks=int(stat[19]),source_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        return result if not inspect.isawaitable(result) else None

    class Middleware:
        def __init__(self,app):
            self.app=app;self.overlay=None;self.lock=asyncio.Lock();self.failure=None;self.generation_lock=asyncio.Lock()
        async def __call__(self,scope,receive,send):
            if scope.get('type')!='http':return await self.app(scope,receive,send)
            if config.get('successor_ref') and scope.get('path')=='/v1/chat/completions':
                async with self.generation_lock:
                    return await self.dispatch(scope,receive,send)
            return await self.dispatch(scope,receive,send)

        async def dispatch(self,scope,receive,send):
            async with self.lock:
                if self.failure is not None and scope.get('path') in ('/metrics','/health'):
                    return await self.app(scope,receive,send)
                require(self.failure is None,'SERVICE_ACTIVATION_FAILED:'+str(self.failure))
                if self.overlay is None:
                    if local and scope.get('path')!='/v1/chat/completions':
                        return await self.app(scope,receive,send)
                    if local:
                        owner_check(local)
                    require(handles,'ACTUAL_API_STATE_NOT_INITIALIZED')
                    if not local:
                        from phase_process import check_window
                    pending=ServiceOverlay(self.app,**handles,output_root=config['overlay_root'],
                        endpoint=config['endpoint'],service_identity=config['service_identity'],
                        owner_check=(lambda:owner_check(local)) if local else lambda:check_window(config['execution_ref']))
                    if local:
                        pending.expected_base_model_path=local['model']
                        pending.bounded_candidate=local['bounded_candidate']
                    else:
                        execution=doc(config['execution_ref'])
                        request=doc(execution['request_ref'])
                        pending.expected_base_model_path=doc(request['input_ref'])['native']['model_path']
                    try:
                        precision=config['named_rpc_overlay'].get('precision_ref')
                        if local:
                            await install_precision(pending.engine,config,local)
                        elif precision:
                            spec=doc(precision)
                            rows=await pending.engine.collective_rpc('omni_production_precision',timeout=120,args=(spec,))
                            require(len(rows)==4 and all(row==rows[0] for row in rows),'four-rank production precision differs')
                            from io_utils import save
                            save(Path(config['overlay_root'])/'PRECISION.json',dict(specification_ref=precision,workers=rows,actual_device_execution=True))
                        # Precision/adapter changes invalidate cached activations.
                        # The owner has drained the lane; never abort live work.
                        reset=await pending.engine.reset_prefix_cache(reset_running_requests=False)
                        require(reset is True,'RL_POLICY_PREFIX_CACHE_RESET_FAILED')
                        from io_utils import save
                        save(Path(config['overlay_root'])/'CACHE-POLICY.json',dict(
                            schema='rl-rollout-cache-policy-v1',kv_cache=True,prefix_cache=True,
                            policy_ref=config['policy_ref'],precision_ref=precision,
                            reset_on_activation=True,reset_succeeded=reset,
                            reuse_scope='same_policy_precision_adapter_and_input'))
                        await pending.activate(config['policy_ref'],config['profile_ref'])
                    except BaseException as error:
                        self.failure=repr(error)
                        from io_utils import save
                        save(Path(config['overlay_root'])/'ACTIVATION-FAILED.json',dict(error=self.failure))
                        raise
                    self.overlay=pending
                elif (local and config.get('successor_ref') and scope.get('path')=='/v1/chat/completions'
                      and self.overlay.active['policy_ref']!=config['successor_ref']
                      and (Path(config['overlay_root'])/'SWITCH-REQUEST.json').exists()):
                    try:
                        marker=doc(pin(Path(config['overlay_root'])/'SWITCH-REQUEST.json'))
                        require(marker==dict(previous=config['policy_ref'],successor=config['successor_ref']),
                                'unexpected owner switch identities')
                        await self.overlay.replace_checkpoint(config['successor_ref'],config['profile_ref'],
                                                              config['named_rpc_overlay']['precision_ref'])
                    except BaseException as error:
                        self.failure=repr(error)
                        raise
            return await self.overlay(scope,receive,send)

    def build(*args,**kwargs):
        return build_with_overlay(original_build,Middleware,*args,**kwargs)

    api_server.init_app_state=init
    api_server.build_app=build
    parsed=parse_service_args(cli,argv)
    from engine_api import engine_kwargs
    parsed.lora_dtype=engine_kwargs(dict(lora_dtype=parsed.lora_dtype,max_loras=parsed.max_loras,
        fully_sharded_loras=parsed.fully_sharded_loras,worker_extension_cls=parsed.worker_extension_cls))['lora_dtype']
    import uvloop
    uvloop.run(api_server.run_server(parsed))


if __name__=='__main__':main()
