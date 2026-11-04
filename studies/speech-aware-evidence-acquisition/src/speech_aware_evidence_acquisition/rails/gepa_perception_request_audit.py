"""Actual independent P→S provenance for a verified Perception GEPA bundle."""
import copy
import json
from .node_prompt_candidate import sha, text_sha
from .tfrl_bank_request_audit import checked_control
ARMS=('parent','child','aa','lock')


def _instruction(request,template):
    control=checked_control(request)
    instruction=control['messages'][1]['content']
    suffix='\n\nNative episode limits: '
    if not instruction.startswith(template+suffix):
        raise ValueError('actual instruction differs from verified node template')
    limits=instruction[len(template+suffix):]
    if json.dumps(json.loads(limits),sort_keys=True)!=limits:
        raise ValueError('noncanonical native limits')
    return control,limits


def _project_p(request,template):
    control,limits=_instruction(request,template)
    if control.get('role_id') != 'perception':
        raise ValueError('Perception request role differs')
    result=copy.deepcopy(request);control=copy.deepcopy(control)
    if any(h.get('role')=='agent_state' for h in request['history']):
        raise ValueError('Perception input unexpectedly consumes an earlier agent state')
    result.pop('request_id');control.pop('contact_id')
    control['messages'][1]['content']='<verified-template>\n'+limits
    history=result['history'];envelope=history[-1]
    for key in ('content','content_sha256','contact_id'):envelope.pop(key)
    envelope['verified_control']=control
    for index,source in enumerate(result['evidence_provenance']):
        if source.get('lock_key')=='rails-supply':
            if source['relpath']!=f"rails-message/{request['request_id']}/{index}":
                raise ValueError('perception supply provenance ownership differs')
            source['relpath']=f'rails-message/<request>/{index}'
    return result


def verify_inputs(sample_id,requests,responses,*,templates,solver_template):
    """Requests map each arm to P/S primary contacts selected by sealed receipt.

    Caller verifies the native bundle/branch receipts and complete output rows.
    This additionally proves actual P inputs differ only by editable template,
    S keeps its instruction/decoding and consumes its own P. Conditional S
    retrieval may differ legitimately when the P candidate set changes.
    """
    if set(requests)!=set(ARMS) or set(responses)!=set(ARMS) or set(templates)!=set(ARMS):
        raise ValueError('four exact Perception arms required')
    projections={};solver_invariants={};ids=[];notes={}
    for arm in ARMS:
        p,s=requests[arm]['perception'],requests[arm]['solver']
        if p['sample_id']!=sample_id or s['sample_id']!=sample_id:
            raise ValueError('request belongs to another sample')
        ids.extend([p['request_id'],s['request_id']])
        projections[arm]=_project_p(p,templates[arm])
        text=responses[arm]['text'];digest=text_sha(text)
        if digest!=responses[arm]['response_text_sha256']:
            raise ValueError('P response text custody differs')
        states=[h for h in s['history'] if h.get('role')=='agent_state' and h.get('state_kind')=='phase:perception']
        if len(states)!=1 or states[0].get('source_response_sha256')!=digest or states[0].get('content_sha256')!=text_sha(states[0]['content']):
            raise ValueError('Solver must consume its own exact P response')
        control,limits=_instruction(s,solver_template)
        if control.get('role_id') != 'solver':
            raise ValueError('Solver request role differs')
        solver_invariants[arm]={key:s.get(key) for key in ('sample_id','audio_seconds','raw_speech_query',
            'media_relpath','speech_ref','sample_rate_hz','decoding_params','task_instruction','tool_schemas')}
        solver_invariants[arm].update(role_id=control['role_id'],limits=limits)
        notes[arm]=digest
    if len(set(ids))!=8:raise ValueError('eight distinct primary contacts required; P cannot be shared')
    if any(projections[a]!=projections['lock'] for a in ARMS):
        raise ValueError('P inputs differ outside declared template')
    if any(solver_invariants[a]!=solver_invariants['lock'] for a in ARMS):
        raise ValueError('frozen Solver/audio/decoding differs')
    return {'schema':'saea-gepa-perception-input-audit-v1','sample_id':sample_id,
        'perception_input_sha256':sha(projections['lock']),'solver_frozen_sha256':sha(solver_invariants['lock']),
        'perception_response_sha256':notes,'primary_contacts':8,'new_model_calls':0,
        'solver_supply':'conditionally recomputed from each own P; not required byte-equal'}
