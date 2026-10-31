"""Offline c2 request replay using actual frozen supply and recorded P/S replies.

No server/gold contacts. The production graph and renderer rebuild requests;
Recorded responses are released only after exact model-wire comparison and
explicitly recorded, narrowly checked state-custody projection.
All scratch accounting stays under a fresh wire-check run. Never registers a run.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def canonical(value): return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(repo, data, recipe_path, experiment, lock_path, out, limit, session_path):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment, ExecutionPlan
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.flight.runtime import WireCheckGate
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.config import protocol_hash, effective_request_decoding_params
    from speech_aware_evidence_acquisition.core.model.transport import LlamaServerTransport
    from speech_aware_evidence_acquisition.core.model.vllm_transport import vllm_request_body
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
    from speech_aware_evidence_acquisition.rails.core_client import FrozenCoreClient
    from speech_aware_evidence_acquisition.rails.phase_runner import run_continuous_program
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data / 'work/codex').resolve()):
        raise ValueError('fresh external output required')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    recipe = json.loads(recipe_path.read_bytes())
    checkpoint_path = (data / recipe['checkpoint']['relpath']).resolve()
    if not checkpoint_path.is_relative_to(data.resolve()):
        raise ValueError('checkpoint escapes data root')
    checkpoint = read_checkpoint(checkpoint_path, expected_sha256=recipe['checkpoint']['sha256'])
    if len(checkpoint.rows) != recipe['rows']:
        raise ValueError('checkpoint row count differs from recipe')
    selected = checkpoint.rows if limit is None else checkpoint.rows[:limit]
    if not selected: raise ValueError('empty selection')
    expected = {}; counts = {}; wire_counts = {}; historical_bodies = []; delivered = {}; custody_deltas = []; mutex = threading.Lock()
    exp=json.loads((repo/'configs/experiment'/f'{experiment}.json').read_bytes());cell,=exp['cells']
    model_path=repo/'configs/model'/f"{cell['model_fragment']}.json"
    model=json.loads(model_path.read_bytes())
    model_name=model['served_model_name']
    if model['transport'] != 'vllm':
        raise ValueError('this proof requires vLLM')
    class OfflineRenderer(LlamaServerTransport):
        # The historical endpoint is not contacted. Media/projection/shape validation
        # is inherited; this subclass can render but can NEVER send.
        def _assert_session_still_valid(self): pass
        def __call__(self, request_bytes): raise RuntimeError('offline renderer cannot send')
    session = SessionReceipt.from_json(json.loads(session_path.read_bytes()))
    renderer = OfflineRenderer(f'http://{session.host}:{session.port}', data,
        session, {}, data / 'work/scratch/runtime-root',
        prompt_order=model['prompt_order'],
        reasoning_content_guard=model['reasoning_content_guard'])
    for row in selected:
        if row.keep_defect() is not None: raise ValueError('invalid checkpoint row')
        output, = row.output_rows; ep=output['native_episode_receipt']
        branch, = [b for b in ep['perception_branches']['branches'] if b['branch_id']=='lock']
        contacts={c['agent_contact_id']:c for v in ep['role_visits'] for c in v['execution_receipt']['contacts']}
        trace={json.loads(line)['record_id']:json.loads(line)['payload'] for line in row.trace_lines}
        own=[]
        for field in ('contact_id','solver_contact_id'):
            contact=contacts[branch[field]];rid=contact['request_id'];request=trace[rid+'/request']['payload']
            response=trace[rid+'/response']; cost=trace[rid+'/cost']
            wire_response=trace[rid+'/wire-response']
            if (trace[rid+'/request']['payload_sha256'] != contact['request_sha256']
                    or hashlib.sha256(canonical(request)).hexdigest() != contact['request_sha256']):
                raise ValueError('historical request custody differs')
            body=renderer.render_request(canonical(request));wire=vllm_request_body(body,model_name)
            key=hashlib.sha256(canonical(wire)).hexdigest();tokens=cost['prompt_tokens']
            if key in wire_counts and wire_counts[key] != tokens: raise ValueError('inconsistent historical token count')
            wire_counts[key]=tokens
            historical_bodies.append(wire)
            own.append({'request':request,'boundary':boundary_identity([request]),'wire':canonical(wire),
                        'text':response['text'],'tokens':tokens,'cost':cost,'wire_response':wire_response})
        expected[row.sample_id]=own;counts[row.sample_id]=0;delivered[row.sample_id]=output['text']
    def token_counter(body):
        key=hashlib.sha256(canonical(vllm_request_body(json.loads(body), model_name))).hexdigest()
        if key not in wire_counts:
            current = vllm_request_body(json.loads(body), model_name)
            changed = [[k for k in set(current) | set(old) if current.get(k) != old.get(k)] for old in historical_bodies]
            fields = min(changed, key=len)
            raise ValueError('new rendered request differs before historical FIT reuse: ' + ','.join(sorted(fields)))
        return wire_counts[key]
    class Replay:
        render_request=renderer.render_request
        def __call__(self, raw):
            payload=json.loads(raw);sid=payload['sample_id']
            with mutex:
                index=counts[sid]
                if index>=2: raise ValueError('unexpected extra contact')
                old=expected[sid][index]
                body=renderer.render_request(raw)
                if canonical(vllm_request_body(body,model_name)) != old['wire']:
                    raise ValueError('vLLM wire body differs')
                compared = copy.deepcopy(payload)
                if boundary_identity([compared]) != old['boundary']:
                    # A c2 run must retain its truthful provenance. These two
                    # state-custody fields are validated by the renderer but
                    # do not enter its messages. Record the transition; never
                    # rewrite the runtime payload or claim raw boundary equality.
                    current_history = compared.get('history', [])
                    old_history = old['request'].get('history', [])
                    if index != 1 or not current_history or len(current_history) != len(old_history):
                        raise ValueError('unexpected boundary change outside solver custody')
                    current_state, old_state = current_history[0], old_history[0]
                    if (current_state['producer_policy_id'] != 'phase-native-module-v1-c2'
                            or old_state['producer_policy_id'] != 'phase-native-module-v1-c17'):
                        raise ValueError('unexpected state producer transition')
                    delta = {'sample_id':sid}
                    for field in ('producer_policy_id', 'source_query_state_sha256'):
                        delta[field] = {'old':old_state[field], 'new':current_state[field]}
                        current_state[field] = old_state[field]
                    if boundary_identity([compared]) != old['boundary']:
                        raise ValueError('request boundary differs beyond recorded state custody')
                    custody_deltas.append(delta)
                counts[sid]+=1
            wire_response = old['wire_response']
            return canonical({'text':old['text'], 'usage':{k: old['cost'][k] for k in ('prompt_tokens','completion_tokens')},
                'wire_response_b64':wire_response['response_b64'],
                'wire_response_sha256':wire_response['response_sha256'],
                'wire_finish_reason':wire_response['finish_reason'],
                'finish_reason':wire_response['finish_reason']})
    ids=set(expected)
    def selected_runner(**kwargs):
        kwargs['flown_sample_ids']=tuple(sorted(ids))
        kwargs['speech_packet_bindings']={k:v for k,v in kwargs['speech_packet_bindings'].items() if k in ids}
        for name in ('bootstrap_facet_ids', 'bootstrap_value_ids', 'initial_feedback_states'):
            if isinstance(kwargs.get(name), Mapping):
                kwargs[name] = {k: v for k, v in kwargs[name].items() if k in ids}
        return run_continuous_program(**kwargs)
    out.mkdir(parents=True)
    publish_new(out/'intent.json',canonical({'checkpoint_sha256':recipe['checkpoint']['sha256'], 'rows':len(ids),'new_model_calls':0}))
    runtime=build_flight_runtime(repo,data,model=cell['model_fragment'],dataset=cell['dataset_fragment'],
        baseline=cell['baseline_fragment'],experiment=experiment,run_id='SAEA-WIRE-CHECK-LOCK-REPLAY-'+out.name.upper(),
        attempt_id='attempt-'+out.name,runtime_root=data/'work/scratch/runtime-root',lock_path=lock_path,
        token_counter=token_counter,continuous_phase_runner=selected_runner)
    client=None
    try:
        values,carrier,split,identity=runtime._prepare()
        if checkpoint.header.get('split_identity_hash') != identity:
            raise ValueError('checkpoint split identity differs')
        from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
        from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
        native=load_native_episode_config(values, default_phase_program(values['policy']))
        if native.max_contacts != 2 or native.prompt_contact_limits != 'legacy-c17-lock-v1':
            raise ValueError('requires reviewed two-contact compatibility mode')
        declared=values['execution_plan']
        from speech_aware_evidence_acquisition.core.reasoning_knobs import fallback_contacts_per_row
        ceiling = 2 + fallback_contacts_per_row(values['policy'], effective_request_decoding_params(values))
        plan=ExecutionPlan(run_id=runtime._run_id,attempt_id=runtime._attempt_id,
            execution_profile=declared['execution_profile'],carrier_lock_key=carrier,split_role=declared['split_role'],
            split_identity_hash=identity,planned_model_calls=ceiling*len(ids),planned_gpu_hours=declared['planned_gpu_hours'],
            planned_speech_audio_seconds=declared['planned_speech_audio_seconds'],protocol_hash=protocol_hash(runtime._composed.config_hash, runtime._arm(), identity)).validate()
        replay=Replay();runtime._plan=plan;runtime._transport=replay
        runtime._prepare_native_phase_runtime(values['policy'])
        client=FrozenCoreClient(**runtime.frozen_core_client_kwargs(plan=plan,gate=WireCheckGate(),transport=replay,run_dir=out/'scratch-client'))
        runtime._client=client
        result=runtime._dispatch_phase_program(client,plan,values['policy'],model_free_check=True)
        if set(counts.values())!={2}: raise ValueError('incomplete replay')
        rows, contact_count, _, _, _ = result
        canonical_rows = runtime._canonical_output_rows(client, output_rows=rows)
        if contact_count != 2 * len(ids) or {r['sample_id']: r['text'] for r in canonical_rows} != delivered:
            raise ValueError('replayed delivery differs from recorded lock output')
        report={'schema':'saea-lock-actual-request-replay-v1','rows':len(ids),'contacts':sum(counts.values()),
            'checkpoint_sha256':recipe['checkpoint']['sha256'],'raw_boundary_equal':not custody_deltas,'boundary_equal_after_custody_projection':True,'vllm_wire_equal':True,'delivered_output_equal':True,
            'historical_session_sha256':sha(session_path),'model_sha256':sha(model_path),
            'protocol_hash':plan.protocol_hash,'split_identity_hash':identity,
            'new_model_calls':0,'planned_contact_ceiling_per_row':ceiling,'capacity_measured':False,'registered':False,
            'limitation':'Recorded response replay; historical token counts; no new inference or serving equivalence claim.'}
        custody_bytes=canonical(custody_deltas)
        report['custody_deltas_sha256']=hashlib.sha256(custody_bytes).hexdigest()
        report['custody_transition_rows']=len(custody_deltas)
        publish_new(out/'custody-deltas.json',custody_bytes)
        publish_new(out/'complete.json',canonical(report));return report
    except Exception as error:
        publish_new(out/'failed.json',canonical({'error':str(error),'automatic_retry':False}));raise
    finally:
        if client is not None:client.close()
        from speech_aware_evidence_acquisition.core.flight.runtime import abort_attempt_if_opened
        abort_attempt_if_opened(data,runtime._run_id,runtime._attempt_id,reason='offline recorded-response replay; no model contacts')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','data','recipe','lock','out','session'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--experiment',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();print(json.dumps(run(a.repo,a.data,a.recipe,a.experiment,a.lock,a.out,a.limit,a.session)))
