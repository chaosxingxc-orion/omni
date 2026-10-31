"""Register and execute R01C CPU encoding through existing isolated custody.

Preserve the frozen encoder, four batch-one workers and16 threads per worker.
Only CPU placement/monitoring and distinct output identities change. No TFRL
iteration or answering-core generation is authorized by this preparation.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import r01c_encoding_write_registration as draft
from fold_a_encoding_admission import clean_head
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

RUN='SAEA-SLURP-R01C-SPEECH-QUERY-CPU-ISOLATED-V1'
INVENTORY='coordination/omni-embed-nemotron-3b-cpu-isolated-r01c-v1.json'
PROTOCOL='configs/preparation/slurp-r01c-speech-query-cpu-isolated-v1.json'
REGISTRATION='docs/readiness/2026-10-31-r01c-cpu-overlap-v1/registration-v1.json'
SESSION='local-nemotron-cpu-isolated-r01c-v1'
CPUS=list(range(192,256))
read,sha,raw=draft.read,draft.sha,draft.predecessor.canonical


def unspent(data):
    for key in ('attempts','preparation-runs','prepared','runs'):
        if (data/key/RUN).exists():raise ValueError('CPU identity already used; inspect without replay')


def derive(repo,data,lock):
    outputs=draft.derive(repo,data,lock)
    base=json.loads(outputs[draft.PROTOCOL]);original=json.loads(outputs[draft.REGISTRATION])
    old_inventory=Path(base['encoder']['path'].replace('${SAEA_DATA_ROOT}',str(data)))
    encoder=copy.deepcopy(read(old_inventory));encoder.update(first_core=192)
    profiles={name:sha(repo/'configs/deployment'/(name+'.json')) for name in (
        'qwen3-omni-real500-2xtp4-s96-b2048-v1','qwen3-omni-real500-2xtp4-s128-b2048-v1',
        'qwen3-omni-real500-1xtp8-s256-b2048-v1','qwen3-omni-real500-1xtp8-s384-b2048-v1')}
    protocol=copy.deepcopy(base)
    protocol.update(encoder={'path':'${SAEA_DATA_ROOT}/'+INVENTORY,'sha256':draft.predecessor.sha256_bytes(raw(encoder))},
        monitor_interval_seconds=600,output_relpath=f'prepared/{RUN}/index.json',
        query_rows_relpath=f'prepared/{RUN}/queries.jsonl',cache_relpath=f'prepared/{RUN}/span-cache',
        cpu_isolation={'schema':'saea-cpu-preparation-isolation-v1','cpu_ids':CPUS,'generation_overlap':True},
        cpu_execution={'schema':'saea-isolated-cpu-preparation-v1','cpuset':'192-255','driver_cpuset':'32-63','serving_profiles':profiles},
        contact_policy='CPU-only fp32 causal four workers x16 threads on192-255, no network/devices, separate existing custody; no automatic retry')
    entry=copy.deepcopy(original['plans'][0]);digest=draft.predecessor.sha256_bytes(raw(protocol))
    entry['execution_plan'].update(run_id=RUN,protocol_hash=digest)
    entry.update(session_id=SESSION,protocol={'relpath':PROTOCOL,'sha256':digest},
                 output_relpath=protocol['output_relpath'],query_rows_relpath=protocol['query_rows_relpath'])
    fields=entry['ledger_row_cells'];fields[1]=RUN;fields[3]='Isolated CPU R01C encoding';fields[13]=REGISTRATION
    fields[14]=f'Root countersigned under {draft.AUTHORITY}, coordinator Codex. Protocol SHA256 {digest}; CPU192-255 separate custody; actual NPU hours0; 2h is wall cap, no retry or implicit downstream adoption.'
    doc={'schema':'saea-prospective-tool-registration-v1','status':'REGISTERED-BEFORE-CONTACT',
        'authority':{'relpath':draft.AUTHORITY,'sha256':sha(repo/draft.AUTHORITY)},'plans':[entry],
        'umbrella_lock':{'path':str(lock),'sha256':sha(lock)},'source_inventory':base['sources'],
        'source_encoder':{'path':str(old_inventory),'sha256':sha(old_inventory)},
        'cpu_isolation':protocol['cpu_isolation'],'actual_npu_hours':0,'downstream_adopted':False}
    return encoder,protocol,doc


def register(repo,data,lock):
    unspent(data);head=clean_head(repo)
    encoder,protocol,doc=derive(repo,data,lock)
    ledger=repo/'docs/exposure-ledger.md'
    if any(p.exists() for p in (repo/PROTOCOL,repo/REGISTRATION,data/INVENTORY)) or '| '+RUN+' |' in ledger.read_text():
        raise ValueError('existing CPU registration; preserve partial work')
    doc['registration_base_commit']=head
    publish_new(data/INVENTORY,raw(encoder))
    for p,d in ((repo/PROTOCOL,protocol),(repo/REGISTRATION,doc)):
        publish_new(p,raw(d));p.chmod(0o644)
    columns=draft.predecessor.LEDGER_COLUMNS
    with ledger.open('a') as f:f.write('\n## Isolated R01C CPU preparation\n\n| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*15)+' |\n| '+' | '.join(doc['plans'][0]['ledger_row_cells'])+' |\n')
    return {'registered':True,'rows':625,'model_contacts':0,'actual_npu_hours':0}


def config(repo,data,lock):
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.contracts.tool_plan import assert_registered_tool_plan
    from speech_aware_evidence_acquisition.core.cpu_preparation_lease import validate_cpu_partition
    from speech_aware_evidence_acquisition.core.speech_preparation_runtime import preparation_lease_root,_cpu_speech_backend
    unspent(data);head=clean_head(repo)
    encoder,protocol,expected=derive(repo,data,lock);actual=read(repo/REGISTRATION)
    if {k:v for k,v in actual.items() if k!='registration_base_commit'}!=expected:
        raise ValueError('registered CPU derivation changed')
    if (repo/PROTOCOL).read_bytes()!=raw(protocol) or (data/INVENTORY).read_bytes()!=raw(encoder):
        raise ValueError('registered CPU bytes changed')
    entry=actual['plans'][0];plan=ExecutionPlan(**entry['execution_plan'])
    validate_cpu_partition(encoder,protocol,repo)
    preparation_lease_root(protocol,encoder,plan,data,_cpu_speech_backend,'speech')
    assert_registered_tool_plan(source_root=repo,exposure_ledger=repo/'docs/exposure-ledger.md',
        umbrella_lock=lock,umbrella_lock_sha256=sha(lock),protocol_path=repo/PROTOCOL,
        protocol_sha256=sha(repo/PROTOCOL),plan=plan)
    return {'plan':entry['execution_plan'],'protocol':{'path':str(repo/PROTOCOL),'sha256':sha(repo/PROTOCOL)},
        'umbrella_lock':actual['umbrella_lock'],'session_id':SESSION,'study_head':head}


def main():
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','config'])
    for key in ('repo','data','lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--out',type=Path);a=p.parse_args()
    if a.stage=='register':result=register(a.repo,a.data,a.lock)
    else:
        if a.out is None or not a.out.resolve().is_relative_to((a.data/'work/codex').resolve()):raise ValueError('owned external config required')
        result=config(a.repo,a.data,a.lock);publish_new(a.out,raw(result));result={'config_sha256':sha(a.out),'rows':625,'model_contacts':0}
    print(json.dumps(result))

if __name__=='__main__':main()
