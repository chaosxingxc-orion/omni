"""Verify protocol-only paired request bytes; no answer repair or model contact."""
import copy
import json
import hashlib
from collections.abc import Mapping
from .slurp_solver_protocol import response_format


def canonical_request(payload):
    value=copy.deepcopy(dict(payload));value.pop('request_id',None)
    decoding=value.get('decoding_params')
    if not isinstance(decoding,dict):raise ValueError('protocol audit requires decoding payload')
    fmt=decoding.pop('response_format',None)
    for message in value.get('history',[]):
        content=message.get('content')
        if not isinstance(content,str):continue
        try:control=json.loads(content)
        except (ValueError,TypeError):continue
        if message.get('role')=='control' and isinstance(control,dict) and control.get('schema')=='saea-native-framework-contact-v1':
            if message.get('contact_id')!=control.get('contact_id') or message.get('content_sha256')!=hashlib.sha256(content.encode()).hexdigest():
                raise ValueError('control identity/hash mismatch')
            control.pop('contact_id');message.pop('contact_id');message.pop('content_sha256')
            message['content']=json.dumps(control,sort_keys=True,separators=(',',':'))
    return value,fmt


def verify_pair(constrained,lock):
    a,fmt=canonical_request(constrained);b,plain=canonical_request(lock)
    if a!=b or plain is not None or not isinstance(fmt,Mapping):
        raise ValueError('protocol pair differs beyond response_format/contact identity: '+str([k for k in set(a)|set(b) if a.get(k)!=b.get(k)]))
    try:
        branches=fmt['json_schema']['schema']['oneOf']
        actions={x['properties']['scenario']['const']:x['properties']['action']['enum'] for x in branches}
        types=branches[0]['properties']['entities']['items']['properties']['type']['enum']
        if fmt!=response_format(actions,types):raise ValueError('protocol schema is not label-only')
    except (KeyError,TypeError,IndexError) as error:
        raise ValueError('malformed protocol schema') from error
    return {'same_input_except_protocol':True,'format_name':fmt['json_schema']['name']}


def verify_contact_pair(trace_records,contacts):
    """Compare original requests and verify any receipted answer-only re-ask.

    One arm may need fallback while the other does not; compare original
    policies, not the outcome-dependent final answer budget.
    """
    from ..core.reasoning_knobs import answer_only_decoding
    originals=[]
    for contact in contacts:
        final=trace_records[f"{contact['request_id']}/request"]['payload']
        fallback=trace_records.get(contact['agent_contact_id']+'/reasoning-fallback')
        if fallback is None:
            originals.append(final);continue
        if (fallback.get('schema')!='saea-thinking-answer-fallback-v1'
                or fallback.get('agent_contact_id')!=contact['agent_contact_id']
                or fallback.get('reask_request_id')!=contact['request_id']):
            raise ValueError('protocol fallback custody mismatch')
        original=trace_records[f"{fallback['first_request_id']}/request"]['payload']
        a,fmt=canonical_request(original);b,endfmt=canonical_request(final)
        a['decoding_params']=answer_only_decoding(a['decoding_params'])
        if a!=b or fmt!=endfmt:
            raise ValueError('protocol fallback changed inputs or output format')
        originals.append(original)
    if len(originals)!=2:raise ValueError('exact paired contacts required')
    return verify_pair(*originals)
