"""Load the two pinned original S1 modules for packet-only binary-mark parsing."""
import hashlib
import sys
import types
from pathlib import Path


def make_parser(refs):
    names={'slot_stages.py','slot_stage_knowledge.py'}
    if set(refs)!=names:raise ValueError('two original S1 parser members required')
    blobs={}
    for name,ref in refs.items():
        path=Path(ref['path']); data=path.read_bytes()
        if path.name!=name or hashlib.sha256(data).hexdigest()!=ref['sha256']:
            raise ValueError('changed S1 parser member: '+name)
        blobs[name]=data
    # Isolate relative imports from any already-loaded business Session package.
    scope='_omni_s1_'+hashlib.sha256(b''.join(blobs[k] for k in sorted(names))).hexdigest()
    pkg=types.ModuleType(scope);pkg.__path__=[]
    sys.modules[scope]=pkg
    for k in ('slot_stage_knowledge.py','slot_stages.py'):
        name=scope+'.'+Path(k).stem
        module=types.ModuleType(name);module.__package__=scope;module.__file__=str(refs[k]['path'])
        sys.modules[name]=module
        exec(compile(blobs[k],module.__file__,'exec'),module.__dict__)
    cls=module.StageRails
    parser=cls.__new__(cls).after_model_call
    def parse(packet,reply):
        if (packet['stage']!='S1' or packet.get('s1_candidate_presentation')!='binary_marking_v1'
            or reply.get('request_id')!=packet['request_id'] or reply.get('wire_sha256')!=packet['wire_sha256']
            or (reply.get('receipt') or {}).get('request_sha256')!=packet['wire_sha256']):
            raise ValueError('binary S1 original packet/reply binding required')
        return parser(packet,reply)
    return parse
