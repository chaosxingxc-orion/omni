"""Explicit N-sample adapter for existing frozen TRAIN partitions; original runner."""
import copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,lines,sha,require,canonical,validate_packet,load
import run as original
# Fixed independent native runner version, original frozen module stays intact.
original.runner=load('rolling_train_native_v1',Path(__file__).parent/'train-runtime-v1/run_s_native.py')
original.original_outcome=original.runner.observation_outcome
original.runner.observation_outcome=original.branch_outcome
original.runner.validate_receipt=original.validate_fit
from speech_aware_evidence_acquisition.rails import candidate_handoff as hb

def preflight(config,workspace):
    cfg=read(config);require(Path(workspace).resolve()==ROOT,'workspace identity')
    require(cfg['schema']=='s-single-intent-native-v1' and cfg['concurrency'] in (64,96,128),'explicit TRAIN concurrency64/96/128')
    require(cfg['dataset_partition'] in ['S_fit','eligible_remainder'],'TRAIN tuning partitions only')
    require(cfg['full_denominator']*3==cfg['expected_rows'],'N x3 explicit')
    require(sha((ROOT/cfg['manifest']).read_bytes())==cfg['manifest_sha256'],'manifest pin')
    parent=read(ROOT/cfg['parent_config'])
    for path,pin in parent['pins'].items():require(sha((ROOT/path).read_bytes())==pin,'parent source pin '+path)
    snapshot=read(ROOT/cfg['snapshot']);require(snapshot['snapshot_sha256']==cfg['snapshot_sha256'],'snapshot identity')
    proofs={r['sample_id']:r for r in lines(ROOT/cfg['prepared']/'source-proof.private.jsonl')}
    require(len(proofs)==cfg['full_denominator'],'N source proofs')
    originals={}
    for sid,proof in proofs.items():
        raw=(ROOT/proof['original_packet']).read_bytes()
        require(sha(raw)==proof['original_packet_sha256'],'original TRAIN packet custody')
        originals[sid]=__import__('json').loads(raw)
    values={v['value_id']:v['content'] for v in snapshot['values']};packets=[]
    for row in lines(ROOT/cfg['manifest']):
        raw=(ROOT/row['payload']).read_bytes();require(sha(raw)==row['packet_sha256'],'packet custody')
        packet=__import__('json').loads(raw)
        require(packet['sample_id']==row['sample_id'] and packet['source_sample_id']==row['source_sample_id'],'identity')
        require(packet['arm']==row['arm']==cfg['arm'],'arm identity')
        proof=proofs[row['source_sample_id']];origin=originals[row['source_sample_id']]
        require(row['P_response_sha256']==proof['P_response_sha256']==origin['handoff']['source_response_sha256']
                ==packet['handoff']['source_response_sha256'],'real P response custody')
        require(row['P_wire_sha256']==proof['P_wire_sha256'],'real P wire custody')
        require(packet['candidate_ids']==[origin['candidate_ids'][row['candidate_index']]],'original P order')
        require(packet['audio_sha256']==origin['audio_sha256'] and packet['wire']['messages'][1]['content'][0]
                ==origin['wire']['messages'][1]['content'][0],'original audio custody')
        require(row['reviewed_hold']==proof['reviewed_hold'] and row['teacher_update_allowed']==(not proof['reviewed_hold']),
                'teacher hold isolation')
        validate_packet(packet,selected=row['candidate_ids'],expected_audio=packet['wire']['messages'][1]['content'][0],
            snapshot=cfg['snapshot_sha256'],values=values,handoff_module=hb)
        original.runner.mapped_wire(packet,row);packets.append((row,packet))
    require(len(packets)==len({r['sample_id'] for r,_ in packets})==cfg['expected_rows'],'N unique branches')
    require(len({r['source_sample_id'] for r,_ in packets})==cfg['full_denominator'],'N unique samples')
    return cfg,packets

original.preflight=preflight
original.runner.preflight=preflight
if __name__=='__main__':original.main()
