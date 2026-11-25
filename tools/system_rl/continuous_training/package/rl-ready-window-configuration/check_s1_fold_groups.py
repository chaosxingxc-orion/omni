"""Read rendered S1 packets through the original fresh sampling and offline label paths."""
import argparse
import json
import sys
from pathlib import Path
from build import pin,save
from prepare_s1_recordings import checked,needed,read


def check(inputs,release,profile_ref,scope,out):
    tree=release/'tree/parents';sys.path.insert(0,str(tree/'fresh-cohort-successor'))
    import cohort_io
    capture=cohort_io.load(Path(__file__).parent/'source/cohort_capture.py','scope_cohort_capture')
    contract=cohort_io.load(tree/'rollout-capture-contract/capture.py','scope_capture_contract')
    profile=read(checked(profile_ref));contract.validate_profile(profile)
    data=read(inputs);panel=read(checked(data['panel_ref']));pages=read(checked(panel['sampling']['pages']))
    domains={d['domain']:[json.loads(line) for line in checked(d['metadata']).open()]
             for d in read(checked(pages['configuration']['dataset']))['domains']}
    expected=read(scope)
    needed(expected['trainable_mask']==dict(P=False,S1=True,S2=False,D=False),'different phase')
    entries={e['reference_id']:e for e in data['rows']};request_ids=set()
    for row in panel['draws']:
        rid=row['reference_id'];d=entries.pop(rid);p=read(checked(d['packet']))
        domain,line,_=rid.split(':'); raw=domains[domain][int(line)-1]
        qid=p['sample_id']; gid='acceptance:'+rid
        annotation=dict(raw,recording_ref_id=rid,source_record_sha256=cohort_io.digest(raw))
        labs=capture.labels_for('S1',dict(type_universe=p['allowed_types']),annotation)
        needed(labs==d['supervision'] and d['fold']==p['branch']==row['fold'],'gold/fold projection differs')
        previous=None
        for k in range(expected['replicates']):
            identity=dict(group_id=gid,episode_id=gid+':'+str(k),node='S1',fold=d['fold'],attempt_id=gid,
                    replicate=k,seed=k,parents=[],run_id='cpu-fold-consumption',boot_id='no-service',query_id=qid,split='TRAIN')
            q=capture.sample_packet(p,profile,identity,contract)
            needed(q['request_id'] not in request_ids and q['wire']['chat_template_kwargs']['enable_thinking'] is False,
                   'duplicate request or thinking enabled')
            content=cohort_io.digest(q['wire']['messages'])
            needed(previous in (None,content) and q['original_packet']['wire_sha256']==p['wire_sha256'],'conditioning changed')
            request_ids.add(q['request_id']);previous=content
    needed(not entries and len(request_ids)==len(data['rows'])*expected['replicates'],'incomplete G groups')
    return save(out,dict(config=pin(scope),inputs=pin(inputs),sampling_profile=profile_ref,protocol=pin(contract.__file__),
        checked_recordings=len(data['rows']),distinct_fresh_request_identities=len(request_ids),
        model_calls=0,optimizer_steps=0,fields='same fold and messages across G; official offline labels on supplied types only'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('release',type=Path)
    p.add_argument('profile',type=Path);p.add_argument('scope',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(check(a.inputs,a.release,pin(a.profile),a.scope,a.output)))
