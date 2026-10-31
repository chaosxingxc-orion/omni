"""Audit overlap of historical A0 checkpoints and the fixed real500 roster.

This inventory makes no model/scorer contacts and grants no resume permission.
An overlap is a potential reusable observation, not a validated scientific merge.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def partition(parent_ids,historical_ids,engineering_ids):
    parent=set(parent_ids);old=set(historical_ids);new=set(engineering_ids)
    if len(parent)!=len(parent_ids) or len(old)!=len(historical_ids) or len(new)!=len(engineering_ids):raise ValueError('duplicate sample identity')
    if not old<=parent or not new<=parent:raise ValueError('cohort outside full baseline population')
    groups={'historical':sorted(old),'engineering_overlaps_historical':sorted(new&old),
            'engineering_additional_if_compatible':sorted(new-old),'remaining_if_all_engineering_complete_and_compatible':sorted(parent-old-new),
            'remaining_without_engineering_reuse':sorted(parent-old)}
    if old&(new-old) or (old|new)&(parent-old-new) or old|(new-old)|(parent-old-new)!=parent:raise ValueError('invalid disjoint partition')
    return groups


def audit(repo,data,recipe,roster,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
    assert_execution_environment()
    read=json.loads;registration=read(recipe.read_bytes());pin=registration['checkpoint'];path=(data/pin['relpath']).resolve()
    if not path.is_relative_to(data.resolve()):raise ValueError('checkpoint escapes data root')
    checkpoint=read_checkpoint(path,expected_sha256=pin['sha256'])
    if len(checkpoint.rows)!=registration['rows'] or any(r.keep_defect() for r in checkpoint.rows):raise ValueError('historical cohort incomplete')
    source=repo/'configs/dataset/slurp-speech-native-evo-fold-b-1080.json';parent=read(source.read_bytes());engine=read(roster.read_bytes())
    expected=split_identity_hash(prefixed('slurp',[x['sample_id'] for x in parent['items']]))
    if parent['split_identity_hash']!=expected or engine['parent_split_identity_hash']!=expected:raise ValueError('parent population identity differs')
    ids=[x['sample_id'] for x in engine['items']]
    if engine['count']!=500 or split_identity_hash(prefixed('slurp',ids))!=engine['split_identity_hash']:raise ValueError('engineering roster differs')
    groups=partition([x['sample_id'] for x in parent['items']],[r.sample_id for r in checkpoint.rows],ids)
    proofs={}
    for name,values in groups.items():proofs[name]={'count':len(values),'ids_sha256':hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()}
    result={'schema':'saea-a0-roster-coverage-audit-v1','parent_count':len(parent['items']),'parent_split_identity_hash':expected,
        'historical_checkpoint_sha256':checkpoint.sha256,'engineering_split_identity_hash':engine['split_identity_hash'],'groups':proofs,
        'max_potential_covered_after_one_engineering_candidate':len(groups['historical'])+len(groups['engineering_additional_if_compatible']),
        'new_model_calls':0,'new_scoring_calls':0,'scientific_reuse_authorized_by_this_audit':False,
        'selection_rule':'Historical complete rows first; first engineering candidate only for missing IDs after exact method/input/custody verification. Never select by answer quality or fastest completion.',
        'required_before_reuse':['Engineering candidate fully sealed and scored','Exact model/prompt/evidence method compatibility including preparation and decoding','Explicit source/projection registration and trace custody','All historical and engineering costs retained','One official full1080 manifest score; never average subset F1']}
    out.mkdir(exist_ok=False)
    for name,doc in [('cohorts.json',groups),('complete.json',result)]:
        with (out/name).open('x') as f:json.dump(doc,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','recipe','roster','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.repo,a.data,a.recipe,a.roster,a.out)))
