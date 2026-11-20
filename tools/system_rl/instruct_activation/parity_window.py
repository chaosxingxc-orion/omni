"""Compile, execute or verify the same immutable policy-switch experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


def read(p):
    return json.loads(Path(p).read_bytes())


def write(p, v):
    with p.open('x') as f:
        json.dump(v, f, indent=2, allow_nan=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(c):
    if c['schema']!='omni-lora-parity-window-v1':raise ValueError('window schema')
    stages=c['stages'];lanes=c['lanes'];names=[s['name'] for s in stages]
    if (not lanes or len(set(lanes))!=len(lanes) or len(set(names))!=len(names)
        or not all(re.fullmatch('[a-z][a-z0-9_-]*',n) for n in [*lanes,*names])):raise ValueError('stage/lane names')
    states=[s['policy'] for s in stages]
    if (not states or states[0]!='initial' or states[-1]!='successor' or
        states!=sorted(states) or set(states)!={'initial','successor'}):raise ValueError('expected one directed policy transition')
    if any(type(s[k])!=int or s[k]<0 for s in stages for k in ('group','replicate')):raise ValueError('sample selectors')
    lim=c['limits']
    if set(lim)!={'token','sequence','max_argmax_flips'} or any(type(lim[k]) not in (int,float) or not math.isfinite(lim[k]) or lim[k]<0 for k in lim) or type(lim['max_argmax_flips'])!=int:raise ValueError('invalid parity limits')
    if type(c['training_world_size'])!=int or c['training_world_size']<states.count('successor')*len(lanes):raise ValueError('too many requests for one scoring window; split the window')
    if type(c['timeout_seconds'])!=int or c['timeout_seconds']<=0:raise ValueError('request timeout')


def pinned(ref):
    if sha(ref['path'])!=ref['sha256']:raise ValueError('sealed input changed')
    return read(ref['path'])


def compile_window(template, profile, out):
    from config_profile import resolve
    t=read(template);p=read(profile);c=resolve(t,p);validate(c)
    release=Path(c['release']);rows=read(c['manifest'])['rows']
    lanes=[];guard=None;refs=None
    for lane in c['lanes']:
        cfg=read(release/f'lane-{lane}/config.json')
        server=read(release/f'bundle/SERVER-{lane}.json')
        ids={k:server[k] for k in ('policy_ref','successor_ref')}
        if refs is not None and (ids!=refs or guard!=cfg['guard']):raise ValueError('lane guard/policies differ')
        refs,guard=ids,cfg['guard'];owner=server['service_identity']['container_name']
        attempt=owner.removeprefix(cfg['name']+'-')
        if owner==attempt:raise ValueError('lane container identity')
        lanes.append(dict(name=lane,endpoint=server['endpoint'].removesuffix('/v1/chat/completions'),container=owner,
                          world_size=len(cfg['cards']),policy_state=str(Path(cfg['root'])/'runs'/attempt/'state/policy')))
    if len({Path(l['policy_state']).parents[4] for l in lanes})!=1:raise ValueError('lane release roots differ')
    if len({l['endpoint'] for l in lanes})!=len(lanes):raise ValueError('duplicate endpoint')
    out.mkdir(mode=0o700);write(out/'TEMPLATE.json',t);write(out/'PROFILE.local.json',p);write(out/'RESOLVED.json',c)
    previous=None
    for stage in c['stages']:
        row,=[r for r in rows if all(r[k]==stage[k] for k in ('group','replicate'))]
        name=stage['name'];sw=stage['policy']=='successor';ref=refs['successor_ref' if sw else 'policy_ref']
        if sha(ref['path'])!=ref['sha256']:raise ValueError('policy changed')
        policy=read(ref['path']);wire=out/f'wire-{name}.json';write(wire,row['packet']['wire'])
        remote=Path(read(release/f'lane-{lanes[0]["name"]}/config.json')['root']).parent
        cp_ref=pinned(policy['publication_ref'])['checkpoint_ref']
        cp=pinned(cp_ref)
        if sw and cp_ref['sha256']!=sha(c['checkpoint']):raise ValueError('unexpected successor checkpoint')
        tensor_count=len(cp['resident_identity']['publication']['tensors'])
        settings=dict(output=str(remote/name/'activation-probe01'),guard=guard,
          wire=dict(path=str(remote/wire.name),bytes=wire.stat().st_size,sha256=sha(wire)),
          lanes=[dict(l,successor=sw,policy_state=l['policy_state']+('/successor' if sw else '')) for l in lanes],
          adapter_sha256=policy['assets'][0]['sha256'],adapter_tensors=tensor_count,request_timeout_seconds=c['timeout_seconds'])
        if sw and previous and previous['policy']=='initial':
            settings['switch']=dict(drained_path=str(remote/previous['name']/'activation-probe01/DRAINED.json'),
                                   policies=dict(previous=refs['policy_ref'],successor=refs['successor_ref']))
        write(out/f'{name}.json',settings);previous=stage


def review(c):
    validate(c);root=Path(c['evidence']);hf=root/'hf';samples=read(c['manifest'])['rows']; limits=c['limits']
    receipt=read(hf/'EXIT.json');world=c['training_world_size']
    if receipt['exit_code']!=0 or receipt['complete_ranks']!=world:raise ValueError('incomplete HF')
    expected=sha(c['checkpoint']);checkpoint=read(c['checkpoint']);records=[];specs=[];prior={}
    tensors=checkpoint['resident_identity']['publication']['tensors']
    artifact=checkpoint['ranks'][0]['files']['lora_adapter/adapter_model.safetensors']
    if not tensors or world!=checkpoint['world_size']:raise ValueError('unexpected checkpoint roster')
    for stage in c['stages']:
        n=stage['name'];phase=root/n/'activation-probe01';drained=read(phase/'DRAINED.json')
        if not drained['guards_releasable'] or list(phase.rglob('UNKNOWN.json')):raise ValueError('not drained')
        sample,=[r for r in samples if all(r[k]==stage[k] for k in ('group','replicate'))]
        canonical=json.dumps(sample['packet']['wire'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        receipts=dict(zip(c['lanes'],read(phase/'RESULT.json')['lanes'],strict=True))
        for lane in c['lanes']:
            response=read(phase/lane/'RESPONSE.json')
            if response['status']!=200 or hashlib.sha256(response['raw'].encode()).hexdigest()!=response['response_sha256'] or sha(phase/lane/'RESPONSE.json')!=receipts[lane]['receipt_ref']['sha256']:
                raise ValueError('invalid terminal response')
            if stage['policy']=='initial':prior[lane]=receipts[lane]['activation_ref']['sha256']
            if read(phase/lane/'INTENT.json')['wire_sha256']!=hashlib.sha256(canonical).hexdigest():raise ValueError('sample differs')
        if stage['policy']=='successor':
            paired=read(root/n/'PAIRED.json')['rows']
            if [x['lane'] for x in paired]!=c['lanes']:raise ValueError('unexpected lanes')
            for x in paired:
                if x['prompt']!=sample['prompt_token_ids'] or read(phase/x['lane']/'RESPONSE.json')['response_sha256']!=x['response']:raise ValueError('response binding')
                switch=read(root/'states'/x['lane']/'SWITCHED.json');act=root/'states'/x['lane']/'ACTIVATION.json'
                if (not switch['old_removed_from_engine'] or not switch['cache_reset'] or switch['old_id']==switch['new_id']
                    or switch['activation_ref']['sha256']!=sha(act) or read(act)['weights_ref']['sha256']!=x['adapter_sha']):raise ValueError('policy switch')
                lane=x['lane'];state=act.parent;cache=read(state/'CACHE-POLICY.json')
                precision=read(state/'PRECISION.json');activation=read(act)
                ranks=range(len(activation['service_identity']['physical_devices']))
                receipt=receipts[lane]
                if (not ranks or any(sha(state/file)!=receipt[key]['sha256'] for file,key in (
                        ('ACTIVATION.json','activation_ref'),('PRECISION.json','precision_ref'),('CACHE-POLICY.json','cache_ref')))
                    or not all(cache[k] is True for k in ('reset_succeeded','kv_cache','prefix_cache'))
                    or cache['previous_activation_ref']['sha256']!=prior[lane] or cache['policy_ref']!=activation['policy_ref']
                    or cache['precision_ref']!=precision['specification_ref'] or not precision['actual_device_execution']
                    or switch['new_id']!=activation['adapter_id'] or x['adapter_sha']!=artifact
                    or any(sorted(w['rank'] for w in record['workers'])!=list(ranks) for record in (activation,precision))
                    or any(w['adapter_id']!=switch['new_id'] or w['sha256']!=artifact or
                           w['compared_original_keys']!=len(tensors) or len(w['loaded_tensors'])!=len(tensors)
                           for w in activation['workers'])):raise ValueError('rank or cache proof')
                records.append(x);specs.append(n)
    packet=read(root/'PAIRED.json')
    if records!=packet['rows'] or ('limits' in packet and limits!=packet['limits']):raise ValueError('scored roster or criterion changed')
    results=[]
    for rank in range(world):
        x=records[rank%len(records)];row=read(hf/f'rank-{rank}/RESULT.json')
        if (row['rank']!=rank or row['checkpoint_sha']!=expected or row['response_sha']!=x['response'] or row['observed_policy_artifact']!=x['adapter_sha'] or row['lane']!=x['lane'] or row['tokens']!=len(x['tokens']) or row['optimizer_updates']!=0):raise ValueError('HF result identity')
        results.append(all(math.isfinite(row[p+'_max_abs']) and row[p+'_max_abs']<=limits['token'] and
                           math.isfinite(row[p+'_sequence_abs']) and row[p+'_sequence_abs']<=limits['sequence'] and
                           len(row[p+'_top1_flips'])<=limits['max_argmax_flips'] for p in ('train','eval')))
    return dict(passed=all(results),unique_recordings=len({x['packet']['sample_id'] for x in records}),
                requests=len(records),actions=sum(len(x['tokens']) for x in records),ranks=world,
                stages=specs,limits=limits,checkpoint_sha256=expected)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);cmd=parser.add_subparsers(dest='action',required=True)
    compile_=cmd.add_parser('compile')
    for key in ('template','profile','output'):compile_.add_argument('--'+key,type=Path,required=True)
    for action in ('run','pack','review'):
        cmd.add_parser(action).add_argument('--window',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='compile':compile_window(args.template,args.profile,args.output)
    else:
        c=read(args.window/'RESOLVED.json');validate(c)
        if args.action=='run':
            from check_policy import check
            for stage in c['stages']:check(read(args.window/(stage['name']+'.json')))
        elif args.action=='pack':
            from updated_http_parity import prepare
            root=Path(c['evidence']);joined=[]
            for stage in c['stages']:
                if stage['policy']!='successor':continue
                part=root/stage['name'];out=part/'PAIRED.json'
                prepare(part,args.window/f"wire-{stage['name']}.json",Path(c['manifest']),out,lanes=c['lanes'],
                        expected_tensors=read(args.window/(stage['name']+'.json'))['adapter_tensors'])
                joined.extend(read(out)['rows'])
            write(root/'PAIRED.json',dict(rows=joined,training_use=False,limits=c['limits']))
        else:
            report=review(c);print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
