"""Restore a pinned task-serving identity under the shared guard; no engine restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def restore(repo, data, backup, out, expected_sha256):
    sys.path[:0]=[str(repo/'src'),str(repo/'scripts/analysis')]
    from tfgrpo_recovery_queue import exclusive,quiescence
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    from speech_aware_evidence_acquisition.core.config import read_fragment,resolve_runtime_receipt_path
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    read=lambda p:json.loads(p.read_bytes())
    raw=lambda d:(json.dumps(d,sort_keys=True,indent=2)+'\n').encode()
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('fresh owned transition output required')
    names=['serving-tfgrpo-ctx48k-resident-v1','saea-real500-tp4-i0','saea-real500-tp4-i1']
    files={str(Path(n)/leaf):(backup/n/leaf).read_bytes() for n in names for leaf in ('session.json','session-receipt.json')}
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    if hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=expected_sha256:
        raise ValueError('backup identity changed')
    model=read_fragment(repo,'model','bf16-thinking-real500-tp4-s128-v1')
    receipt=read(resolve_runtime_receipt_path(repo,model))
    if model['decoding_params']['max_tokens']!=4096:raise ValueError('task4096 model required')
    for n in names:
        if read(backup/n/'session.json')['serving_geometry']['max_tokens']!=4096:raise ValueError('task4096 backup required')
    for n in names[1:]:verify_vllm_session(SessionReceipt.from_json(read(backup/n/'session-receipt.json')),receipt)
    with exclusive(data/'locks/generation.guard'):
        if (data/'locks/flight.lock').exists():raise ValueError('active generation lease')
        quiet=quiescence([8200,8201])
        for n in names:
            if read(data/'serving'/n/'session.json')['serving_geometry']['max_tokens']!=16384:
                raise ValueError('expected learning16384 current identity')
        out.mkdir(parents=True)
        for rel in files:
            p=out/'original'/rel;p.parent.mkdir(parents=True,exist_ok=True)
            publish_new(p,(data/'serving'/rel).read_bytes())
        publish_new(out/'intent.json',raw({'backup_sha256':expected_sha256,'quiescence':quiet,'no_engine_restart':True}))
        for rel,blob in files.items():
            target=data/'serving'/rel;temp=target.with_name(target.name+'.task-restore-new')
            publish_new(temp,blob);os.replace(temp,target)
            fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        for n in names[1:]:verify_vllm_session(SessionReceipt.from_json(read(data/'serving'/n/'session-receipt.json')),receipt)
        publish_new(out/'complete.json',raw({'max_tokens':4096,'reasoning_tokens':3072,'new_model_calls':0,'no_engine_restart':True}))
    return {'restored':True,'new_model_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','backup','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    print(json.dumps(restore(**vars(p.parse_args()))))
