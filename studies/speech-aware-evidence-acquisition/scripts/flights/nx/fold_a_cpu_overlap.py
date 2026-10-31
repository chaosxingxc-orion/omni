"""Prepare Fold A retrieval vectors on disjoint CPUs while generation continues.

This is a separately registered CPU tool run with its own output/cache namespace.
It does not replace the queued encoder or make its outputs eligible for inference;
a verified downstream adoption is still required before those vectors are used.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
import fold_a_encoding_admission as prior
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.contracts.tool_plan import assert_registered_tool_plan

RUN = 'SAEA-SLURP-FOLD-A-SPEECH-QUERY-CPU-ISOLATED-V1'
INVENTORY = 'coordination/omni-embed-nemotron-3b-cpu-isolated-fold-a-v1.json'
PROTOCOL = 'configs/preparation/slurp-fold-a-speech-query-cpu-isolated-v1.json'
REGISTRATION = 'docs/readiness/2026-10-31-fold-a-cpu-overlap-v1/registration-v1.json'
SESSION = 'local-nemotron-cpu-isolated-fold-a-v1'
CPUS = list(range(32, 64))
read, sha, raw = prior.read, prior.sha, prior.draft.canonical


def unspent(data):
    for directory in ('attempts', 'preparation-runs', 'prepared', 'runs'):
        if (data/directory/RUN).exists():
            raise ValueError('isolated encoder already started; inspect without replay')


def derive(repo, data, lock):
    original, base = prior.validated_inputs(repo, data, lock)
    old_inventory = Path(base['encoder']['path'].replace('${SAEA_DATA_ROOT}', str(data)))
    encoder = copy.deepcopy(read(old_inventory))
    encoder.update(workers=2, first_core=32)
    protocol = copy.deepcopy(base)
    protocol.update(encoder={'path': '${SAEA_DATA_ROOT}/'+INVENTORY,
                            'sha256': prior.draft.sha256_bytes(raw(encoder))},
        cpu_workers=2, output_relpath=f'prepared/{RUN}/index.json',
        query_rows_relpath=f'prepared/{RUN}/queries.jsonl', cache_relpath=f'prepared/{RUN}/span-cache',
        cpu_isolation={'schema': 'saea-cpu-preparation-isolation-v1', 'cpu_ids': CPUS, 'generation_overlap': True},
        contact_policy='Two CPU fp32 workers, each batch-one with 16 threads; disjoint CPUs32-63, own cache, no accelerator or network. No automatic retry.')
    entry = copy.deepcopy(original)
    digest = prior.draft.sha256_bytes(raw(protocol))
    entry['execution_plan'].update(run_id=RUN, protocol_hash=digest)
    entry.update(protocol={'relpath':PROTOCOL,'sha256':digest}, session_id=SESSION,
        output_relpath=protocol['output_relpath'], query_rows_relpath=protocol['query_rows_relpath'])
    entry.pop('runtime_config_shape', None)
    cells=entry['ledger_row_cells']
    cells[0],cells[1],cells[3],cells[13]='2026-10-31',RUN,'Independent isolated CPU Fold A retrieval preparation',REGISTRATION
    cells[4]='Frozen CPU fp32 Omni-Embed-Nemotron-3B; two workers x16 threads; no answering core'
    cells[14]=(f'Registered before contact by root under {prior.AUTHORITY}. Protocol SHA256 {digest}. '
        'Overlap is CPU-only on32-63 with separate custody and outputs; no model-method or data change. '
        'planned_gpu_hours=2 is a wall-time ceiling, actual NPU hours=0. No automatic replay; no inference adoption by this record.')
    document={'schema':'saea-prospective-tool-registration-v1','status':'REGISTERED-BEFORE-CONTACT',
        'authority':{'relpath':prior.AUTHORITY,'sha256':sha(repo/prior.AUTHORITY)},
        'plans':[entry], 'umbrella_lock':{'path':str(lock),'sha256':sha(lock)},
        'source_protocol':{'relpath':prior.draft.FOLD_A_FRAGMENT,'sha256':sha(repo/prior.draft.FOLD_A_FRAGMENT)},
        'source_encoder':{'path':str(old_inventory),'sha256':sha(old_inventory)},
        'cpu_isolation':protocol['cpu_isolation'],'actual_npu_hours':0,'claims_system_lock':False,
        'downstream_adoption':'Not implied; rebind consumers only after exact source and output validation.'}
    return encoder, protocol, document


def register(repo, data, lock):
    unspent(data);head=prior.clean_head(repo)
    encoder,protocol,document=derive(repo,data,lock)
    ledger=repo/prior.LEDGER;before=ledger.read_bytes()
    if any(p.exists() for p in (repo/PROTOCOL,repo/REGISTRATION,data/INVENTORY)) or ('| '+RUN+' |').encode() in before:
        raise ValueError('isolated registration already exists; preserve partial evidence')
    document.update(registered_at_utc=datetime.now(timezone.utc).isoformat(),registration_base_commit=head)
    publish_new(data/INVENTORY,raw(encoder))
    for path,value in ((repo/PROTOCOL,protocol),(repo/REGISTRATION,document)):
        publish_new(path,raw(value));path.chmod(0o644)
    if ledger.read_bytes()!=before:raise ValueError('ledger changed')
    cells=document['plans'][0]['ledger_row_cells']
    with ledger.open('a') as f:
        f.write('\n## Isolated Fold A CPU preparation\n\n| '+' | '.join(prior.draft.LEDGER_COLUMNS)+' |\n'
            +'| '+' | '.join(['---']*15)+' |\n| '+' | '.join(cells)+' |\n')
    return {'files':[PROTOCOL,REGISTRATION,prior.LEDGER],'encoder_contacts':0}


def runtime_config(repo,data,lock):
    unspent(data);head=prior.clean_head(repo)
    encoder,protocol,expected=derive(repo,data,lock)
    registered=read(repo/REGISTRATION)
    if (set(registered) != set(expected) | {'registered_at_utc', 'registration_base_commit'}
            or any(registered.get(k)!=value for k,value in expected.items())):
        raise ValueError('registered isolated CPU inputs changed')
    if (repo/PROTOCOL).read_bytes()!=raw(protocol) or (data/INVENTORY).read_bytes()!=raw(encoder):
        raise ValueError('registered CPU protocol or inventory changed')
    entry=registered['plans'][0]
    config={'plan':entry['execution_plan'],'protocol':{'path':str(repo/PROTOCOL),'sha256':sha(repo/PROTOCOL)},
        'umbrella_lock':registered['umbrella_lock'],'session_id':SESSION,'study_head':head}
    assert_registered_tool_plan(source_root=repo,exposure_ledger=repo/prior.LEDGER,
        umbrella_lock=lock,umbrella_lock_sha256=config['umbrella_lock']['sha256'],
        protocol_path=repo/PROTOCOL,protocol_sha256=config['protocol']['sha256'],plan=ExecutionPlan(**config['plan']))
    return config


def execute(repo,data,lock,out):
    from speech_aware_evidence_acquisition.core import speech_preparation_runtime as runtime
    config=runtime_config(repo,data,lock)
    protocol=read(repo/PROTOCOL);encoder=read(data/INVENTORY)
    runtime.preparation_lease_root(protocol,encoder,ExecutionPlan(**config['plan']),data,runtime._cpu_speech_backend,'speech')
    out=out.resolve()
    if not out.is_relative_to(data.resolve()) or out.is_relative_to(repo.resolve()) or out.exists():
        raise ValueError('fresh external coordinator required')
    out.mkdir(parents=True,exist_ok=False)
    pins={str(p):sha(p) for p in (repo/PROTOCOL,repo/REGISTRATION,data/INVENTORY,lock)}
    publish_new(out/'config.json',raw(config))
    publish_new(out/'started.json',raw({'source_commit':config['study_head'],'inputs':pins,
        'run_id':RUN,'cpu_ids':CPUS,'actual_npu_hours':0}))
    argv=[sys.executable,'-m','speech_aware_evidence_acquisition.core.speech_preparation_runtime',
        '--config',str(out/'config.json'),'--config-sha256',sha(out/'config.json'),
        '--data-root',str(data),'--source-root',str(repo)]
    try:
        with (out/'encoding.log').open('x') as log:
            result=subprocess.run(argv,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        publish_new(out/'encoding.exit.json',raw({'exit':result.returncode}))
        if result.returncode:raise ValueError('CPU encoding failed; preserve evidence without replay')
        if prior.clean_head(repo)!=config['study_head'] or any(sha(Path(p))!=h for p,h in pins.items()):
            raise ValueError('source or inputs changed')
        receipt=data/'preparation-runs'/RUN/'encode-v1/receipt.json'
        rec=read(receipt)
        if (rec['status']!='SEALED' or rec['run_id']!=RUN or rec['study_head']!=config['study_head']
                or rec['protocol_sha256']!=sha(repo/PROTOCOL) or rec['cpu_isolation']!=protocol['cpu_isolation']
                or read(data/'attempts'/RUN/'encode-v1.json')['state']!='FINALIZED'):
            raise ValueError('sealed isolated CPU receipt required')
        query=data/protocol['query_rows_relpath'];index=data/protocol['output_relpath']
        if sha(query)!=rec['query_rows_sha256'] or sha(index)!=rec['index_sha256']:
            raise ValueError('sealed CPU output differs')
        from fold_a_retrieval_inputs import verify_rows
        sources=runtime.pinned_json(protocol['sources'])['rows']
        rows=[json.loads(line) for line in query.read_text().splitlines()]
        verify_rows(rows,sources,protocol['dimension'])
        if read(index)['output']['rows']!=rows:raise ValueError('CPU index and query export differ')
        publish_new(out/'complete.json',raw({'source_commit':config['study_head'],'run_id':RUN,'rows':1080,
            'receipt':str(receipt),'receipt_sha256':sha(receipt),'actual_npu_hours':0,
            'downstream_adopted':False,'next':'Bind validated CPU outputs into the retrieval preparation chain.'}))
    except BaseException as error:
        publish_new(out/'failed.json',raw({'reason':str(error),'automatic_retry':False}));raise
    return {'rows':1080,'actual_npu_hours':0}


def main():
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('register','check','run'))
    for k in ('repo-root','data-root','lock-path'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--out',type=Path);a=p.parse_args()
    repo,data,lock=a.repo_root.resolve(),a.data_root.resolve(),a.lock_path.resolve()
    if a.stage=='register':result=register(repo,data,lock)
    elif a.stage=='check':result=runtime_config(repo,data,lock)
    else:
        if a.out is None:p.error('run requires out')
        result=execute(repo,data,lock,a.out)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
