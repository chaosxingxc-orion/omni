"""Classify original H1 checkpoints by actual context integrity, without scoring.

Preserve every original row and cost. Only complete rows with seven unmodified,
FIT-bound contacts can be reused; all other parent rows require repaired work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
RUN='SAEA-SLURP-TFRL-CURRENT-SEED-I1-H1-V1'
CHECKPOINT_SHA='5ac8ce266b26288e05854adde933101959ca43e145971872b910e5dd3352f87d'
NAME='slurp-tfrl-current-seed-i1-h1-v1'


def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())


def context_integrity(output):
    """No answer, score or correctness is consulted by this partition rule."""
    episode=output.get('native_episode_receipt',{})
    cf=episode.get('context_fit')
    if not isinstance(cf,dict) or cf.get('schema')!='saea-native-context-fit-v1':
        raise ValueError('missing original context accounting')
    contacts=cf.get('contacts')
    if not isinstance(contacts,list) or len(contacts)!=7:
        raise ValueError('seven-contact context accounting required')
    ids=[x['contact_id'] for x in contacts]
    if len(set(ids))!=7:raise ValueError('repeated context contact')
    moved=[c for c in contacts if c.get('context_compaction') is not None]
    if cf.get('compacted_contacts')!=len(moved):raise ValueError('context compaction count differs')
    return not moved


def partition(parent_ids, outputs):
    ids=[o['sample_id'] for o in outputs]
    if len(parent_ids)!=len(set(parent_ids)) or len(ids)!=len(set(ids)) or not set(ids)<=set(parent_ids):
        raise ValueError('invalid source population')
    kept=[o['sample_id'] for o in outputs if context_integrity(o)]
    affected=[o['sample_id'] for o in outputs if not context_integrity(o)]
    return {'reusable_ids':sorted(kept),'affected_completed_ids':sorted(affected),
            'remaining_ids':[sid for sid in parent_ids if sid not in set(kept)],
            'uncompleted_ids':[sid for sid in parent_ids if sid not in set(ids)]}


def audit(source,data,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
    from tfrl_seed_request_audit import verify
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned custody output required')
    roster_path=source/'docs/receipts/rosters/slurp-stage3-construction-r01c-h1.json'
    roster=read(roster_path);ids=[r['sample_id'] for r in roster['items']]
    if len(ids)!=313:raise ValueError('full313 required')
    cp_path=data/'runs'/RUN/'generation-v1'/(RUN+'.rows.ckpt.jsonl')
    cp=read_checkpoint(cp_path,expected_sha256=CHECKPOINT_SHA)
    if len(cp.rows)!=25 or cp.torn_tail_bytes or any(r.keep_defect() for r in cp.rows):raise ValueError('original25 custody differs')
    if read(data/'attempts'/RUN/'generation-v1.json')['state']!='ABORTED':raise ValueError('original attempt not terminal')
    outputs=[r.output_rows[0] for r in cp.rows];parts=partition(ids,outputs)
    plan=read(source/'docs/readiness'/NAME/'sampling-plan.json')
    partial_plan={**plan,'row_plan':[r for r in plan['row_plan'] if r['row_id'] in parts['reusable_ids']]}
    trace={}
    for row in cp.rows:
        for line in row.trace_lines:
            r=json.loads(line)
            if r['record_id'] in trace and trace[r['record_id']]!=r['payload']:raise ValueError('conflicting trace identity')
            trace[r['record_id']]=r['payload']
    seed_check=verify(partial_plan,[o for o in outputs if o['sample_id'] in parts['reusable_ids']],trace)
    fit_root=data/'prepared'/(RUN+'-FIT')/'fit-v1'
    measured=[];pins=[]
    for row in cp.rows:
        if row.sample_id not in parts['reusable_ids']:continue
        episode=row.output_rows[0]['native_episode_receipt'];limit=episode['context_fit']['prompt_limit_tokens']
        contacts=[c for v in episode['role_visits'] for c in v['execution_receipt']['contacts']]
        if len(contacts)!=7:raise ValueError('complete seven-contact source required')
        for contact in contacts:
            rid=contact['request_id'];req=trace[rid+'/request'];payload=req['payload'];response=trace[rid+'/response']
            if (hashlib.sha256(canonical(payload)).hexdigest()!=contact['request_sha256']
                or req['payload_sha256']!=contact['request_sha256']
                or hashlib.sha256(response['text'].encode()).hexdigest()!=contact['response_text_sha256']):raise ValueError('source contact hash differs')
            identity=hashlib.sha256(canonical({'sample_id':row.sample_id,'agent_contact_id':contact['agent_contact_id']})).hexdigest()
            path=fit_root/(identity+'.request-fit.json');fit=read(path);value=fit['output']
            if (hashlib.sha256(canonical(value)).hexdigest()!=fit['output_sha256']
                or value['boundary_sha256']!=boundary_identity([payload])
                or value['sample_id']!=row.sample_id or value['agent_contact_id']!=contact['agent_contact_id']
                or value['generation_contacts']!=0 or value['every_request_fits'] is not True
                or value['sample_count']!=1):raise ValueError('FIT source binding differs')
            record,=value['rows'];tokens=record['measured_prompt_tokens']
            if (type(tokens)is not int or tokens<0 or tokens>limit or record['measurement']!='vllm-chat-tokenize'
                    or record['fits']is not True or record['request_body_sha256']==''):raise ValueError('repaired exact budget would refuse historical contact')
            measured.append(tokens);pins.append({'relpath':str(path.relative_to(data)),'sha256':sha(path)})
    if (len(parts['reusable_ids']),len(parts['affected_completed_ids']),len(parts['remaining_ids']))!=(3,22,310):raise ValueError('unexpected source integrity partition')
    exp=read(source/f'configs/experiment/{NAME}.json');cell,=exp['cells']
    parent=read(source/f"configs/dataset/{cell['dataset_fragment']}.json")
    selected=[r for r in parent['items'] if r['sample_id'] in set(parts['remaining_ids'])]
    subset={**roster,'items':[r for r in roster['items'] if r['sample_id'] in set(parts['remaining_ids'])],
            'split_identity_hash':split_identity_hash(prefixed('slurp',parts['remaining_ids'])),'count':310}
    report={'schema':'saea-tfrl-h1-context-integrity-custody-v1','source_checkpoint':{'relpath':str(cp_path.relative_to(data)),'sha256':cp.sha256},
        'parent_roster':{'relpath':str(roster_path.relative_to(source)),'sha256':sha(roster_path)},
        'parent_rows':313,'complete_original_rows':25,'reusable_rows':3,'affected_completed_rows':22,'remaining_rows':310,
        'uncompleted_rows':288,'partition':parts,'selection':'Actual context compaction only; no score or answer selection',
        'retained_fit_receipts':pins,'retained_contacts':len(measured),'max_retained_measured_prompt_tokens':max(measured),
        'original_seed_request_audit':seed_check,'original_cost_lower_bound':{k:sum(r.cost.get(k,0) for r in cp.rows) for k in ('calls','prompt_tokens','completion_tokens','audio_seconds')},
        'all_original_artifacts_preserved':True,'new_model_calls':0,'new_tokenizer_calls':0,
        'remaining_requirement':'Replay three untouched rows through repaired graph before final source-union acceptance.'}
    out.mkdir(parents=True)
    for name,v in [('custody.json',report),('remaining-roster.json',subset)]:publish_new(out/name,canonical(v)+b'\n')
    summary={k:v for k,v in report.items() if k not in ('partition','retained_fit_receipts')}
    summary['custody_sha256']=sha(out/'custody.json');summary['remaining_roster_sha256']=sha(out/'remaining-roster.json')
    publish_new(out/'complete.json',canonical(summary)+b'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','data','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(audit(**vars(p.parse_args()))))
