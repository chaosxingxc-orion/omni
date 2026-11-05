#!/usr/bin/env python3
"""Read the resident session's full identity without acquiring or sending requests."""
import argparse
import hashlib
import json
from pathlib import Path
from speech_aware_evidence_acquisition.core.session.receipt import read_session_receipt
from speech_aware_evidence_acquisition.core.session.vllm_identity import assert_vllm_session_alive


def execute(config):
    doc=json.loads(Path(config).read_bytes())
    if set(doc)!={'schema','session','runtime_receipt'} or doc['schema']!='saea-resident-preflight-v1':
        raise ValueError('exact resident identity check required')
    for key in ('session','runtime_receipt'):
        pin=doc[key]
        if (set(pin)!={'path','sha256'} or not Path(pin['path']).is_absolute()
                or Path(pin['path']).is_symlink()
                or hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()!=pin['sha256']):
            raise ValueError('resident source pin differs')
    session=read_session_receipt(Path(doc['session']['path']))
    runtime=json.loads(Path(doc['runtime_receipt']['path']).read_bytes())
    assert_vllm_session_alive(session,runtime)
    return {'schema':doc['schema'],'ok':True,'pid':session.pid,'host':session.host,'port':session.port,
            'inputs':doc,'new_model_calls':0,'fit_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    print(json.dumps(execute(p.parse_args().config),sort_keys=True))
