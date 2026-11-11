"""S1/S2 adapters through existing openJiuwen hooks and source-bound completion.

`transport(packet, contact_id)` is dependency-injected by the existing owner:
validate FIT/admission/lease, journal raw bytes, and return CoreCompletion.
No network provider, automatic retry or background queue is created here.
"""
from __future__ import annotations

import json

from .slot_stages import digest


class NativeStageCaller:
    def __init__(self, transport):
        self.transport = transport

    async def __call__(self, packet, after_model_call):
        from ..reproduction.ojw.native_roles import (
            NativeRoleNode, NativeRoleNodeSpec, NativeRoleContactContext,
            FrozenCoreRoleFiniteCompletionError,
        )
        from .core_client import CoreCompletion
        contact = packet['request_id']
        flags = {}
        completion = None
        outcome = None
        prompt = packet['wire']['messages'][0]['content']
        # Same SDK bridge used by native S: framework history carries the bound
        # user text, while transport sends the exact multimodal packet.
        user = next(p['text'] for p in packet['wire']['messages'][1]['content'] if p['type'] == 'text')

        def before(role, ctx):
            if role != packet['stage'].lower() or flags.get('before'):
                raise ValueError('wrong or repeated before_model_call')
            if digest(packet['wire']) != packet['wire_sha256']:
                raise ValueError('prepared S packet changed')
            flags['before'] = True

        async def metered(request):
            nonlocal completion
            expected = [{'role': 'system', 'content': prompt.strip()}, {'role': 'user', 'content': user}]
            history = [dict(m) for m in request.messages]
            actual = [{k: v for k, v in m.items() if k != 'metadata'} for m in history]
            if (not flags.get('before') or request.tool_schemas or actual != expected
                    or request.context.sample_id != packet['sample_id']
                    or request.context.agent_contact_id != contact):
                raise ValueError('SDK history/source binding mismatch')
            flags['framework_history_sha256'] = digest(history)
            completion = await self.transport(packet, contact)
            if not isinstance(completion, CoreCompletion):
                raise TypeError('existing journal-backed CoreCompletion required')
            completion.assert_binding(sample_id=packet['sample_id'], agent_contact_id=contact,
                                      response_text=completion.text)
            if completion.request_id != packet['request_id'] or completion.request_sha256 != packet['wire_sha256']:
                raise ValueError('completion does not bind prepared wire')
            return completion

        def after(role, ctx):
            nonlocal outcome
            if not flags.get('before') or completion is None:
                raise ValueError('after_model_call without observed completion')
            raw = json.loads(completion.wire_response_bytes)
            outcome = after_model_call(packet, {'text': completion.text,
                'finish_reason': completion.finish_reason, 'usage': raw.get('usage'),
                'receipt': {'response_sha256': completion.response_sha256,
                            'request_sha256': completion.request_sha256}})
            flags['after'] = True

        node = NativeRoleNode(
            spec=NativeRoleNodeSpec(role_id=packet['stage'].lower(), role_kind='slot',
                                    prompt=prompt, max_iterations=1),
            metered_request=metered,
            contact_context=NativeRoleContactContext(sample_id=packet['sample_id'], agent_contact_id=contact,
                evidence_pins={'S_wire': packet['wire_sha256'], 'S_version': packet['version']}),
            candidate_before=before, candidate_after=after)
        try:
            try:
                await node.invoke_direct(user)
            except FrozenCoreRoleFiniteCompletionError:
                # SDK may terminate a truncated generation before the after hook.
                if completion is None:
                    raise
                if not flags.get('after'):
                    after(packet['stage'].lower(), None)
            if not flags.get('after') or outcome is None:
                raise ValueError('native S after_model_call did not run')
            outcome['framework_history_sha256'] = flags['framework_history_sha256']
            return outcome
        finally:
            await node.close()


class JournalStageTransport:
    """Reuse the established observed_response/admission/journal implementation.

    `post(wire)` is the existing synchronous vLLM adapter. `check(packet)` must
    validate prepared request FIT and the owner lease before fresh admission.
    Passing an established owner admission object is mandatory. No queue or
    automatic retries are introduced by this adapter.
    """
    def __init__(self, *, journal, post, observed_response, admission, check, post_packet=None):
        if admission is None or not callable(check):
            raise ValueError('existing owner admission and FIT/lease check required')
        self.journal, self.post = journal, post
        self.observed_response, self.admission, self.check = observed_response, admission, check
        self.post_packet = post_packet

    async def __call__(self, packet, contact):
        import hashlib
        from .core_client import CoreCompletion
        from .slot_stages import canonical
        self.check(packet)
        custody = {'request_id': packet['request_id'], 'sample_id': packet['sample_id'],
                   'history': [{'role': 'control', 'contact_id': contact}],
                   'wire_sha256': packet['wire_sha256'], 'stage': packet['stage'],
                   'branch': packet['branch'], 'version': packet['version']}
        raw, latency, reused = await self.observed_response(
            self.journal, custody,
            lambda: self.post_packet(packet) if self.post_packet is not None else self.post(packet['wire']),
            admission=self.admission)
        obj = json.loads(raw)
        if isinstance(obj, dict) and 'error' in obj:
            raise ValueError('observed HTTP error; retained in contact journal: '
                             + json.dumps(obj['error'], ensure_ascii=False, sort_keys=True))
        choice = obj['choices'][0]
        message = choice['message']
        text = message.get('content')
        if text is None and choice.get('finish_reason') in ('length', 'content_filter'):
            text = ''
        if not isinstance(text, str):
            raise ValueError('observed model response has no string content')
        return CoreCompletion(text=text, request_id=packet['request_id'], sample_id=packet['sample_id'],
            agent_contact_id=contact, request_sha256=packet['wire_sha256'],
            response_sha256=hashlib.sha256(raw).hexdigest(), response_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            tool_calls_json=canonical(message.get('tool_calls')), wire_response_bytes=raw,
            tool_names=(), tool_schemas_sha256=hashlib.sha256(b'[]').hexdigest())
