"""Prepare one pinned, reusable H1 search partition without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def prepare(repo, data, output, *, selection_rows=80, seed=20261102):
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.roster import read_roster
    from speech_aware_evidence_acquisition.data.slurp import load_slurp_train_material
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    from speech_aware_evidence_acquisition.rails.gepa_partition import build_partition
    roster_paths = {h:repo/f'docs/receipts/rosters/slurp-stage3-construction-r01c-{h}.json' for h in ('h1','h2')}
    rosters = {h:read_roster(p) for h,p in roster_paths.items()}
    view_path=repo/'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'
    test_path=repo/'configs/dataset/slurp-speech-native-test-full.json'
    view=json.loads(view_path.read_bytes());test=json.loads(test_path.read_bytes())
    media={r['sample_id']:r['audio_sha256'] for r in view['items']}
    if len(media)!=len(view['items']):raise ValueError('duplicate media identity')
    labels={}
    for r in load_slurp_train_material(load_lock(umbrella_lock_path(repo)),data):
        for recording in r.recordings:
            labels[recording.sample_id]={'utterance_id':f'slurp:train:{int(r.utterance_id):06d}', 'intent':r.scenario+'_'+r.action}
    def rows(half):
        result=[]
        for item in rosters[half]['items']:
            sid=item['sample_id'];label=labels[sid]
            if label['utterance_id']!=item['utterance_id']:raise ValueError('roster utterance mismatch')
            result.append({'sample_id':sid,'utterance_id':label['utterance_id'],'audio_sha256':media[sid],
                           **({'intent':label['intent']} if half=='h1' else {})})
        return result
    train=rows('h1');excluded=rows('h2')
    # Test view is media identity only: no test label/transcript loading.
    for r in test['items']:
        excluded.append({'sample_id':r['sample_id'],'utterance_id':r['sample_id'].rsplit(':',1)[0],
                         'audio_sha256':r['audio_sha256']})
    if (len(train),len(rosters['h2']['items']),len(test['items']))!=(313,312,13078):
        raise ValueError('fixed H1/H2/test population changed')
    result=build_partition(train,excluded,target_selection=selection_rows,seed=seed)
    result['input_files_sha256']={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [*roster_paths.values(),view_path,test_path]}
    raw=(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
    if output.exists():
        if output.read_bytes()!=raw:raise ValueError('existing partition differs; never overwrite')
    else:publish_new(output,raw)
    return {'feedback_rows':len(result['feedback_ids']),'selection_rows':len(result['selection_ids']),
            'partition_sha256':result['partition_sha256'],'file_sha256':hashlib.sha256(raw).hexdigest(),
            'new_model_calls':0,'test_labels_read':False}


if __name__=='__main__':
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--selection-rows',type=int,default=80);p.add_argument('--seed',type=int,default=20261102)
    a=p.parse_args();assert_execution_environment()
    print(json.dumps(prepare(**vars(a))))
