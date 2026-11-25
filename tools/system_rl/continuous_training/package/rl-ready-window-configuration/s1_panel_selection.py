"""Split before observing outcomes; measure and subsample S1 using sealed native replies."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

from build import pin, save
from prepare_s1_recordings import checked, needed, read
from source.config_profile import bind_environment, stage_config
from source.panel_receipts import collection_calls


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def partition(cfg):
    data = read(checked(cfg['inputs']))
    panel = read(checked(data['panel_ref']))
    draws = {d['reference_id']: d for d in panel['draws']}
    n = cfg['holdout_percent']
    needed(data['schema']=='s1-selected-inputs-v1' and not panel['formal_training_set']
           and type(n) is int and 0<n<100 and type(cfg['seed']) is int, 'invalid panel split')
    rows = []
    for e in data['rows']:
        rid = e['reference_id']
        d = draws.pop(rid)
        p = read(checked(e['packet']))
        b = p['source_binding']
        needed(b['split']=='TRAIN' and b['reference_id']==rid and p['stage']=='S1'
               and p['branch']==d['fold']==e['fold'] and b['source_component']
               and p['allowed_types']==e['supervision']['annotated_types'] and d['stratum']==
               '/'.join((b['domain'],p['branch'],'present' if e['supervision']['present_types'] else 'absent'))
               and digest(p['wire'])==p['wire_sha256'], 'split input binding differs')
        check = int(digest([cfg['seed'],b['source_component']]),16)*100 < n*(1<<256)
        rows.append(dict(reference_id=rid, fold=d['fold'], component=b['source_component'],
            split='check' if check else 'selection', stratum=d['stratum'], packet=e['packet'],
            supervision=e['supervision'], panel_conditional_inclusion=d['conditional_inclusion'],
            fold_probability=d['fold_assignment_probability']))
    needed(not draws and {r['split'] for r in rows}=={'check','selection'}, 'incomplete split')
    return dict(schema='s1-panel-partition-v1', inputs=cfg['inputs'], seed=cfg['seed'],
                holdout_percent=n, counts=dict(Counter(r['split'] for r in rows)), rows=rows)


def packet_parser(profile=None):
    from source.s1_reply_parser import make_parser
    root=Path(__file__).resolve().parent.parents[1]
    cfg=read(root/'full-train-source-bridge/CONFIG.json')
    names={'slot_stages.py','slot_stage_knowledge.py'}
    refs={Path(r['path']).name:dict(r,path=str(root.parents[1]/r['path']))
          for r in cfg['code_pins'] if Path(r['path']).name in names}
    if profile is not None:
        needed(isinstance(profile,str) and Path(profile).name==profile and profile.endswith('.json'),
               'invalid S1 parser profile')
        members=read(Path(__file__).resolve().parent/profile)
        needed(members.keys()==refs.keys() and all(members[k]['sha256']==r['sha256'] for k,r in refs.items()),
               'screen and rollout parser versions differ')
    return make_parser(refs)


def classify(packet, supervision, receipt, parse):
    """Read actual response bytes, bound to the original wire, intent and source."""
    env=receipt['native_envelope']
    req=packet['request_id']; wire=packet['wire_sha256']
    needed(packet['stage']==env['stage']=='S1' and packet['s1_candidate_presentation']=='binary_marking_v1'
           and packet['source_binding']['split']=='TRAIN'
           and req==env['request_id']==receipt['request_id'] and not receipt['synthetic']
           and wire==digest(packet['wire'])==env['wire_ref']['sha256']==receipt['wire_sha256'],
           'S1 receipt/wire identity differs')
    checked(env['wire_ref'])
    intent=read(checked(env['intent_ref'])); response=read(checked(env['response_ref']))
    payload=intent['payload']; raw=base64.b64decode(response['response_b64'],validate=True)
    raw_sha=hashlib.sha256(raw).hexdigest()
    needed(intent['schema']=='saea-contact-intent-v1' and response['schema']=='saea-contact-response-v1'
           and intent['identity']==response['identity'] and response['intent_sha256']==env['intent_ref']['sha256']
           and req==intent['request_id']==response['request_id']
           and all(packet[k]==payload[k] for k in ('request_id','stage','branch','sample_id','version','wire_sha256'))
           and response['request_sha256']==intent['request_sha256']==digest(payload)
           and raw_sha==response['response_sha256']==receipt['native_response_sha256']==receipt['response_sha256']
           and receipt['receipt']==dict(request_sha256=wire,response_sha256=raw_sha),'journal binding differs')
    native=json.loads(raw); choice,=native['choices']
    needed(native['model']==packet['wire']['model'] and choice['finish_reason']==receipt['finish_reason']
           and (choice['message']['content'] or '')==receipt['text'],'native output differs')
    needed(supervision['annotated_types']==packet['allowed_types'] and
           set(supervision['present_types'])<=set(packet['allowed_types']), 'supervision outside fold')
    parsed=parse(packet,receipt)
    if parsed['status'] not in ('valid_nonempty','valid_empty'):
        return 'protocol',[],[]
    gold=set(supervision['present_types']); pred=set(parsed['selected_types'])
    miss=sorted(gold-pred); extra=sorted(pred-gold)
    kind=('mixed' if extra else 'miss') if miss else ('extra' if extra else 'correct_positive' if gold else 'correct_empty')
    return kind,miss,extra


def observed(cfg,split):
    parse=packet_parser(cfg.get('screen_parser_profile'))
    targets={}
    for r in split['rows']:
        p=read(checked(r['packet']))
        targets[(p['sample_id'],r['fold'])]=(r,p)
    needed(len(targets)==len(split['rows']), 'ambiguous panel identity')
    found={}; audit=Counter()
    for ref in cfg['collections']:
        seal=read(checked(ref)); v=read(checked(seal['version']))
        needed(seal['immutable'] and seal['marker_scope']=='collection_analysis'
               and v['mode']=='live' and v['status'] in ('SEALED','SEALED_WITH_FAILURES') and not v['synthetic']
               and v['unresolved_requests']==0,'unclosed collection')
        # No historical policy name inference from the model alias alone.
        needed(all(isinstance(cfg[k],str) and len(cfg[k])==64 for k in ('system_version','execution_version'))
               and v['system_version']==cfg['system_version'] and v.get('execution_version')==cfg['execution_version'],
               'different frozen system or execution')
        for call in collection_calls(v):
            if call['stage']!='S1':continue
            audit['S1_calls']+=1
            p=call['packet']; target=targets.get((p['sample_id'],p['branch']))
            if not target:continue
            r,expected=target;rid=r['reference_id']; audit['recording_fold_matches']+=1
            if (p['wire_sha256']!=expected['wire_sha256'] or p['source_binding']!=expected['source_binding']
                or p['wire']!=expected['wire'] or p['allowed_types']!=expected['allowed_types'] or p['s1_candidate_presentation']!=expected['s1_candidate_presentation']):continue
            needed(call['status']=='known' and rid not in found, 'unresolved or repeated matching observation')
            result,miss,extra=classify(p,r['supervision'],call['receipt'],parse)
            found[rid]=dict(outcome=result, missed=miss, extra=extra, receipt=call['receipt']['native_envelope'],
                            seal=ref,source_wire=p['wire_sha256'])
    return found,dict(audit)


def sample(cfg,split,found):
    caps=cfg['cap_per_domain_fold_outcome']
    needed(set(caps)=={'mixed','miss','extra','correct_empty','correct_positive'} and
           all(type(v) is int and v>=0 for v in caps.values()),'invalid outcome quotas')
    eligible_folds={r['fold'] for r in split['rows']}
    active=cfg['training_folds']
    needed(active=='all' or (isinstance(active,list) and active and len(active)==len(set(active))
           and set(active)<=eligible_folds),'unknown/duplicate training fold')
    active=eligible_folds if active=='all' else set(active)
    needed(found.keys()<=set(r['reference_id'] for r in split['rows']), 'unknown response row')
    by_cell=defaultdict(list); counts=Counter(); holdout=[]; strata=defaultdict(Counter)
    for r in split['rows']:
        rid=r['reference_id']; o=found.get(rid); outcome=o['outcome'] if o else 'unobserved'
        domain=r['stratum'].split('/')[0]
        counts['/'.join((r['split'],domain,r['fold'],outcome))]+=1
        strata[r['split']+'/'+r['stratum']][outcome]+=1
        if o and r['split']=='check':
            holdout.append(dict(reference_id=rid,fold=r['fold'],**o))
        if o and outcome in caps and r['split']=='selection' and r['fold'] in active:
            by_cell[(domain,r['fold'],outcome)].append(dict(r,**o))
    complete=len(found)==len(split['rows'])
    selected=[]
    if complete:
        rng=random.Random(cfg['seed'])
        for (_,_,kind),rows in sorted(by_cell.items()):
            k=min(len(rows),caps[kind])
            for row in rng.sample(sorted(rows,key=lambda r:r['reference_id']),k):
                selected.append(dict(row,outcome_conditional_inclusion=k/len(rows)))
        rng.shuffle(selected)
    return dict(schema='s1-outcome-screen-v1',observed=len(found),panel_size=len(split['rows']),
                counts=dict(counts),by_gold_stratum=dict(strata),panel_complete=complete,
                training_folds=sorted(active),selected=selected,check=holdout,
                formal_training_set=False)


def main():
    from multimodal_jev.local_runtime_storage import check_paths
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['split','score'])
    p.add_argument('config');p.add_argument('--profile',required=True);a=p.parse_args()
    site=read(a.profile)
    if site['schema']=='omni-local-profile-v2':site,_=bind_environment(site,os.environ)
    cfg=stage_config(read(a.config),site,'s1-partition-config-v1' if a.action=='split' else 's1-outcome-config-v1');out=Path(cfg['output']);check_paths({'screen':out.parent},min_free_fraction=.10)
    if a.action=='split':
        needed(cfg['schema']=='s1-partition-config-v1','split configuration required')
        result=partition(cfg)
    else:
        needed(cfg['schema']=='s1-outcome-config-v1','score configuration required')
        split=read(checked(cfg['partition']));needed(split['schema']=='s1-panel-partition-v1','partition required')
        found,audit=observed(cfg,split);result=dict(sample(cfg,split,found),audit=audit)
    print(json.dumps(save(out,dict(result,configuration=cfg))))
if __name__=='__main__':main()
