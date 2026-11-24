"""Owner-installed hooks on an EXISTING vLLM service; no server/engine creation.

Owner drains the original lane and holds its original watchdog/guards throughout
activation and sampling. All public business wires keep the registered model.
"""
import contextvars
import hashlib
import inspect
import json
from pathlib import Path
import types
from io_utils import MODEL, canonical, digest, doc, pin, require, save, storage
from protocol import readback

CURRENT = contextvars.ContextVar('omni_iteration_request', default=None)


async def awaited(value):
    return await value if inspect.isawaitable(value) else value


class ServiceOverlay:
    def __init__(self, app, models, chat, engine, *, output_root, endpoint,
                 service_identity, owner_check, fixture=False):
        self.app, self.models, self.chat, self.engine = app, models, chat, engine
        self.root = Path(output_root)
        self.startup_root = self.root
        storage(self.root, 1024**3)
        self.root.mkdir(parents=True, exist_ok=True)
        self.endpoint, self.service_identity = endpoint, service_identity
        self.owner_check, self.fixture = owner_check, fixture
        self.active = None
        self.calls = {}
        self._bind()

    def _bind(self):
        overlay = self
        old_adapters = self.chat._maybe_get_adapters
        old_name = self.models.model_name
        old_generate = self.engine.generate

        def adapters(_chat, request, *args, **kwargs):
            require(overlay.active is not None, 'POLICY_NOT_ACTIVATED')
            require(request.model == MODEL, 'BUSINESS_MODEL_CHANGED')
            # Existing vLLM adapter resolver; temporary name selects the real
            # loaded LoRARequest. Restore the request before any other handler.
            if overlay.active['base']:
                result=old_adapters(request,*args,**kwargs)
                actual=result[0] if isinstance(result,tuple) else result
                require(actual is None,'BASE_MUST_HAVE_NO_ADAPTER')
                return result
            request.model = overlay.active['alias']
            try:
                result = old_adapters(request, *args, **kwargs)
            finally:
                request.model = MODEL
            actual=result[0] if isinstance(result,tuple) else result
            require(actual is not None and actual.lora_int_id == overlay.active['adapter_id'], 'RESOLVED_STALE_ADAPTER')
            return result

        def model_name(_models, lora_request=None):
            if overlay.active and lora_request is not None and lora_request.lora_int_id == overlay.active['adapter_id']:
                return MODEL
            return old_name(lora_request)

        def generate(*args, **kwargs):
            context = CURRENT.get()
            require(context is not None and overlay.active is not None, 'UNWITNESSED_HTTP_GENERATION')
            overlay.owner_check()
            params = kwargs.get('sampling_params', kwargs.get('params', args[1] if len(args) > 1 else None))
            lora = kwargs.get('lora_request')
            require((lora is None) if overlay.active['base'] else
                    (lora is not None and lora.lora_int_id == overlay.active['adapter_id']), 'GENERATE_ADAPTER_MISMATCH')
            if not context['sampled']:
                return old_generate(*args, **kwargs)
            expected = context['profile']['sampling']
            actual = {k: getattr(params, k, None) for k in expected if k != 'watermarking'}
            actual['top_k'] = 0 if actual['top_k'] == -1 else actual['top_k']
            require(actual == {k: v for k, v in expected.items() if k != 'watermarking'}, 'ACTUAL_SAMPLING_PARAMS')
            require(not getattr(params, 'watermarking', False) and
                    not getattr(params, 'logits_processors', None) and
                    not getattr(params, 'structured_outputs', None) and
                    not getattr(params, 'stop', None) and not getattr(params, 'stop_token_ids', None) and
                    not getattr(params, 'allowed_token_ids', None) and not getattr(params, 'bad_words', None) and
                    getattr(params, 'logprobs', None) == (2 if getattr(overlay,'decode_replay',False) else 0), 'TRANSFORMED_DISTRIBUTION')
            context['generate_observations'].append(dict(adapter_id=None if lora is None else lora.lora_int_id, sampling=actual))
            return old_generate(*args, **kwargs)

        self.chat._maybe_get_adapters = types.MethodType(adapters, self.chat)
        self.models.model_name = types.MethodType(model_name, self.models)
        self.engine.generate = generate

    async def activate(self, policy_ref, profile_ref):
        self.owner_check()
        require(not self.calls, 'NEW_OVERLAY_INSTANCE_PER_POLICY')
        policy, profile = doc(policy_ref), doc(profile_ref)
        base=policy['kind']=='base'
        if not self.fixture and not base:
            if getattr(self,'bounded_candidate',False):
                from publication import validate_publication
                published=validate_publication(policy['publication_ref'])
                require(published['status']=='candidate_for_updated_parity' and published['assets']==policy['assets'],
                        'DP8_PARITY_CANDIDATE_REQUIRED')
            else:
                from lineage import publication
                publication(policy['publication_ref'])
        require(policy['model'] == MODEL and policy['thinking'] is False and policy['kind'] in ('checkpoint','base'), 'PUBLISHED_POLICY_REQUIRED')
        model_config = self.engine.model_config
        if not self.fixture:
            require(getattr(model_config,'model',None)==getattr(self,'expected_base_model_path',None) and
                    bool(self.expected_base_model_path), 'ACTUAL_BASE_MODEL_PATH')
        require(model_config.logprobs_mode == profile['engine_required']['logprobs_mode'] == 'raw_logprobs' and
                model_config.generation_config == profile['engine_required']['generation_config'] == 'vllm', 'ACTUAL_ENGINE_CONFIG')
        require(self.chat.default_chat_template_kwargs.get('enable_thinking') is False, 'EFFECTIVE_THINKING_OFF')
        require(getattr(self.chat, 'reasoning_parser', None) is None, 'REASONING_PARSER_DISABLED')
        if base:
            seal=doc(policy['asset_manifest_ref'])
            require(seal['status']=='CLOSED_IMMUTABLE' and seal['assets']==policy['assets'],'BASE_ASSET_MANIFEST')
            self.active=dict(base=True,alias=None,adapter_id=None,weights_ref=None,fixture=self.fixture,
                policy_ref=policy_ref,publication_ref=None,policy_version=policy['version'],endpoint=self.endpoint,
                service_identity=self.service_identity,source_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                workers=[],profile_ref=profile_ref)
            self.activation_ref=save(self.root/'ACTIVATION.json',self.active)
            return self.activation_ref
        weights = next(a for a in policy['assets'] if Path(a['path']).name == 'adapter_model.safetensors')
        # Verify the actual bytes before loading; no copy of a live directory.
        require(pin(weights['path']) == weights, 'CLOSED_ADAPTER_BYTES')
        alias = policy['artifact_version']
        if self.fixture:
            request = types.SimpleNamespace(lora_name=alias, lora_path=str(Path(weights['path']).parent))
        else:
            from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
            request = LoadLoRAAdapterRequest(lora_name=alias, lora_path=str(Path(weights['path']).parent))
        result = await self.models.load_lora_adapter(request, base_model_name=MODEL)
        require(isinstance(result, str) and alias in self.models.lora_requests, 'EXISTING_LOAD_LORA_FAILED')
        loaded = self.models.lora_requests[alias]
        rows = await awaited(self.engine.collective_rpc('omni_sync_worker_readback', timeout=120,
            args=(loaded.lora_int_id, loaded.lora_path, weights['sha256'], self.fixture)))
        readback(rows, loaded.lora_int_id, weights)
        self.active = dict(base=False,alias=alias, adapter_id=loaded.lora_int_id, weights_ref=weights,
            fixture=self.fixture, policy_ref=policy_ref, publication_ref=policy['publication_ref'],
            policy_version=policy['artifact_version'], endpoint=self.endpoint, service_identity=self.service_identity,
            source_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), workers=rows,
            profile_ref=profile_ref)
        self.activation_ref = save(self.root/'ACTIVATION.json', self.active)
        return self.activation_ref

    async def replace_checkpoint(self, successor, profile, precision):
        """Called under the original owner guard and serialized generation boundary."""
        from vllm.entrypoints.serve.lora.protocol import UnloadLoRAAdapterRequest
        from publication import validate_publication
        self.owner_check()
        old=self.active
        require(self.bounded_candidate and old and not old['base'] and not self.calls,'drained local diagnostic required')
        new=doc(successor)
        published=validate_publication(new['publication_ref'])
        require(new['artifact_version']!=old['alias'] and published['assets']==new['assets']
                and published['status']=='candidate_for_updated_parity','invalid successor')
        root=self.root
        ids=await self.engine.list_loras()
        save(root/'SWITCH-INVENTORY.json',dict(engine_type=type(ids).__name__,engine_ids=sorted(ids),api_ids={k:v.lora_int_id for k,v in self.models.lora_requests.items()}))
        require(set(ids)=={old['adapter_id']} and old['alias'] in self.models.lora_requests,'predecessor identity drift')
        save(root/'SWITCH-INTENT.json',dict(previous=self.activation_ref,successor=successor))
        reply=await self.models.unload_lora_adapter(UnloadLoRAAdapterRequest(lora_name=old['alias']))
        require(isinstance(reply,str) and old['alias'] not in self.models.lora_requests,'API alias removal failed')
        ids=set(await self.engine.list_loras())
        require(ids in (set(),{old['adapter_id']}),'unexpected engine adapter after API unload')
        if ids:require(await self.engine.remove_lora(old['adapter_id']),'engine removal failed')
        require(not await self.engine.list_loras(),'stale adapter after unload')
        require(await self.engine.reset_prefix_cache(reset_running_requests=False) is True,'successor reset failed')
        self.root=(self.startup_root/'updates'/new['artifact_version'].split(':')[-1]
                   if getattr(self,'continuous_activation',False) else root/'successor')
        save(self.root/'PRECISION.json',doc(pin(root/'PRECISION.json')))
        save(self.root/'CACHE-POLICY.json',dict(kv_cache=True,prefix_cache=True,reset_succeeded=True,
             policy_ref=successor,precision_ref=precision,previous_activation_ref=self.activation_ref))
        activation=await self.activate(successor,profile)
        require(set(await self.engine.list_loras())=={self.active['adapter_id']} and
                old['adapter_id']!=self.active['adapter_id'],'engine successor identity drift')
        save(self.root/'SWITCHED.json',dict(old_id=old['adapter_id'],new_id=self.active['adapter_id'],
             old_removed_from_engine=True,cache_reset=True,previous_policy_ref=old['policy_ref'],activation_ref=activation))

    def register(self, call_ref):
        """Owner registers exact original candidate before a single journaled POST."""
        self.owner_check()
        call = doc(call_ref)
        require(self.active and call['wire']['model'] == MODEL and digest(call['wire']) == call['wire_sha256'], 'REGISTER_CANDIDATE')
        require(call['wire_sha256'] not in self.calls, 'DUPLICATE_PROBE_NO_REPLAY')
        self.calls[call['wire_sha256']] = call_ref

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http' or scope.get('path') != '/v1/chat/completions':
            return await self.app(scope, receive, send)
        self.owner_check()
        require(self.active is not None, 'POLICY_NOT_ACTIVATED')
        parts = []
        while True:
            message = await receive()
            require(message['type'] == 'http.request', 'HTTP_BODY_DISCONNECTED')
            parts.append(message.get('body', b''))
            if not message.get('more_body', False):
                break
        body = b''.join(parts)
        wire = json.loads(body)
        require(wire['model'] == MODEL and not wire.get('chat_template_kwargs', {}).get('enable_thinking', False), 'HTTP_MODEL_THINKING')
        wire_sha = digest(wire)
        call_ref = self.calls.get(wire_sha)
        registration=self.root/'probes'/(wire_sha+'.json')
        if call_ref is None and registration.exists():
            call_ref=doc(pin(registration))['call_ref']
            require(doc(call_ref)['wire_sha256']==wire_sha,'REGISTERED_WIRE_IDENTITY')
            self.register(call_ref)
        if getattr(self,'decode_replay',False):
            require(call_ref is None and wire.get('return_token_ids') is True and wire.get('top_logprobs')==2
                    and wire.get('logprobs') is True and wire.get('stream') is False
                    and set(wire.get('vllm_xargs') or {})=={'decode_action_ids'}, 'diagnostic replay wires only')
        elif 'decode_action_ids' in (wire.get('vllm_xargs') or {}):
            raise ValueError('forced actions require an isolated replay release')
        if call_ref is not None:
            intent = self.root/('PROBE-INTENT-'+wire_sha+'.json')
            require(not intent.exists(), 'EXISTING_HTTP_PROBE_INTENT_RECONCILE_NO_REPLAY')
            save(intent,dict(call_ref=call_ref,activation_ref=self.activation_ref,automatic_retry=False))
        # All business calls still route to the active adapter. Only registered
        # probe calls publish a witness; ordinary evaluation wires may differ.
        context = dict(profile=doc(self.active['profile_ref']), generate_observations=[], probe=call_ref is not None,
            sampled=call_ref is not None or wire.get('return_token_ids') is True)
        token = CURRENT.set(context)
        chunks, status = [], None
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return dict(type='http.request', body=body, more_body=False)
            # ASGI consumers continue receiving to observe disconnects. Body
            # replay is once-only; later receives belong to the real channel.
            return await receive()

        async def record(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
            if call_ref is not None and message['type'] == 'http.response.body':
                chunks.append(message.get('body', b''))
            # Persist the endpoint observation BEFORE the client can observe a
            # successful final response, eliminating the witness visibility race.
            if call_ref is None:
                await send(message)

        try:
            await self.app(scope, replay, record)
            if call_ref is None:
                return
            if status is not None and status != 200:
                # Keep the actual upstream rejection, not an invented witness
                # error. A recorded HTTP failure is terminal, never acceptance.
                save(self.root/('HTTP-REJECTED-'+wire_sha+'.json'),dict(
                    call_ref=call_ref,status=status,body=b''.join(chunks).decode('utf-8',errors='replace'),
                    response_sha256=hashlib.sha256(b''.join(chunks)).hexdigest(),
                    actual_generate=context['generate_observations'],accepted=False))
                await send(dict(type='http.response.start',status=status,
                                headers=[(b'content-type',b'application/json')]))
                await send(dict(type='http.response.body',body=b''.join(chunks),more_body=False))
                return
            require(status == 200 and len(context['generate_observations']) == 1, 'HTTP_ACTUAL_GENERATE_REQUIRED')
            self.owner_check()
            call = doc(call_ref)
            rows = [] if self.active['base'] else await awaited(self.engine.collective_rpc('omni_sync_worker_readback', timeout=120,
                args=(self.active['adapter_id'], str(Path(self.active['weights_ref']['path']).parent),
                      self.active['weights_ref']['sha256'], self.fixture)))
            if not self.active['base']:
                readback(rows, self.active['adapter_id'], self.active['weights_ref'])
            event = dict(schema='policy-http-observation-v1', activation_ref=self.activation_ref,
                fixture=self.fixture, provenance='synthetic' if self.fixture else 'current_paid',
                request_id=call['request_id'], wire_sha256=wire_sha,
                response_sha256=hashlib.sha256(b''.join(chunks)).hexdigest(),
                sampling_config_hash=call['sampling_config_hash'], model=MODEL, thinking=False,
                endpoint=self.endpoint, service_identity=self.service_identity, source_boot=self.active['source_boot'],
                adapter_id=self.active['adapter_id'], adapter_sha256=None if self.active['base'] else self.active['weights_ref']['sha256'],
                workers=rows, sampling=context['profile']['sampling'], engine_required=context['profile']['engine_required'],
                logprobs_mode=self.engine.model_config.logprobs_mode, untransformed_distribution=True,
                actual_generate=context['generate_observations'])
            save(self.root/('EVENT-'+wire_sha+'.json'), event)
            await send(dict(type='http.response.start', status=status,
                            headers=[(b'content-type', b'application/json')]))
            await send(dict(type='http.response.body', body=b''.join(chunks), more_body=False))
        finally:
            CURRENT.reset(token)
