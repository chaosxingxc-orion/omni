"""Resolve ambiguous HTTP top scores using a sealed, isolated full-vocabulary observation."""
import json
import subprocess
from pathlib import Path

from decode_action_replay import choice
from decode_lineage import changes, snapshot
from io_utils import digest, doc, pin, require, save


def resolve(spec, replay_root, source, panel):
    packet = Path(spec['packet'])
    original = snapshot(replay_root)
    declaration = doc(pin(packet/'ARGMAX-DELTA.json'))
    diff = changes(original, snapshot(packet))
    code = 'parents/jev-system-rl/implementation/decode_action_processor.py'
    require(declaration['parent'] == digest(original) and diff == declaration['changes']
            and not diff['engine'] and not diff['rows'] and set(diff['files']) == {code},
            'observer arithmetic differs outside action processor')
    targets = doc(declaration['observe'])
    cfg = doc(pin(packet/'CONFIG.local.json'))
    service, = cfg['services']
    receipt = doc(pin(cfg['start_receipts'][service['name']]))
    cid = receipt['container_id']
    device = doc(pin(packet/f"lane-{service['name']}"/'config.json'))
    archive = packet/'observer-evidence'
    def capture_record(name, command):
        # First read from the pinned container, later reviews can run after its release.
        path = archive/(name+'.json')
        if not path.exists():
            value = json.loads(subprocess.check_output(command))
            if name == 'identity':
                state, = value
                value = dict(Id=state['Id'], Name=state['Name'], Image=state['Image'],
                             Mounts=[{k:m[k] for k in ('Source','Destination','RW')}
                                     for m in state['Mounts'] if m['Destination']==str(packet/'bundle')])
            save(path, dict(container_id=cid, value=value))
        record = doc(pin(path))
        require(record['container_id'] == cid, 'archived observer identity')
        return record['value']
    state = capture_record('identity',['docker','inspect',cid])
    require(state['Id'] == cid and state['Name'] == '/'+service['container']
            and state['Image'] == device['image'] and receipt['physical_devices'] == device['cards']
            and any(m['Source'] == m['Destination'] == str(packet/'bundle') and not m['RW']
                    for m in state['Mounts']), 'wrong observation container')
    def record_names(key,pos):
        prefix=f'argmax-{key}-{pos}'
        listing=capture_record('list-'+prefix,['docker','exec',cid,'python3','-c',
            'import pathlib,sys,json; print(json.dumps(sorted(p.name for p in pathlib.Path("/runtime/state").glob(sys.argv[1]))))',
            prefix+'-r*-s*.json'])
        require(listing,'missing rank observations')
        return listing
    def capture(path):
        return capture_record(Path(path).stem,['docker','exec',cid,'cat',path])
    rounds = Path(cfg['output'])
    require(doc(pin(rounds/'DRAINED.json'))['guards_releasable'] and
            doc(pin(rounds/'STARTED.json'))['config_ref'] == pin(packet/'CONFIG.local.json') and
            doc(pin(rounds/'RESULT.json'))['passed'],'observer did not drain')
    policy = capture('/runtime/state/policy/ACTIVATION.json')
    require(policy['weights_ref']['sha256'] == panel['adapter_sha'] and
            policy['policy_version'] == policy['alias'] == 'checkpoint-sha256:'+panel['adapter_sha']
            and policy['service_identity']['container_name'] == service['container'],'observer adapter')
    rr, = doc(pin(rounds/'RESULT.json'))['rounds']
    run = doc(rr)
    width = panel['width']
    offset = spec['group_index']*width
    refs = [r['replay'] for r in source['rows'][offset:offset+width]]
    require(offset >= 0 and len(refs) == width and run['input_refs'] == refs and
            cfg['distinct_panels'] == [refs] and cfg['admission'] == 'group_drained'
            and run['concurrency_per_service'] == {service['name']:width}
            and len(run['requests']) == width, 'not the original complete wave')
    winners, observations = {}, []
    for j,r in enumerate(sorted(run['requests'],key=lambda r:r['replicate'])):
        i = offset+j
        require((r['service'],r['group'],r['replicate']) == (service['name'],0,j),'observer roster')
        direct = doc(r['receipt_ref'])
        raw = json.loads(direct['raw'])
        wire = dict(doc(refs[j]),cache_salt=digest([cfg['measurement_id'],run['block']]))
        intent = doc(pin(Path(r['receipt_ref']['path']).with_name('INTENT.json')))
        ids, items = choice(r['receipt_ref'])
        natural = panel['rows'][i]
        require(raw['prompt_token_ids'] == natural['prompt'] and raw['model'] == wire['model']
                and intent['wire_sha256'] == digest(wire) and intent['automatic_retry'] is False
                and ids == natural['tokens'] == wire['vllm_xargs']['decode_action_ids']
                and [x['logprob'] for x in items] == natural['logprobs'], 'observer history or scores changed')
        key = digest([natural['prompt'],ids])
        for pos in targets.get(key,[]):
            by_rank={}
            for filename in record_names(key,pos):
                v=capture('/runtime/state/'+filename)
                rank,step=v['rank'],v['step']
                require(type(rank) is int and type(step) is int and rank>=0 and step>=0
                        and filename==f'argmax-{key}-{pos}-r{rank}-s{step}.json'
                        and v['conditioning']==key and v['position']==pos and v['action']==ids[pos]
                        and v['vocab']==spec['vocab_size'], 'malformed observation identity')
                by_rank.setdefault(rank,{})[step]={k:x for k,x in v.items() if k not in ('rank','step')}
            require(set(by_rank)==set(range(len(device['cards']))), 'missing TP rank')
            steps=by_rank[0]
            require(set(steps)==set(range(len(steps))) and all(series==steps for series in by_rank.values()),
                    'incomplete or inconsistent rank observations')
            # Chunked prefill can invoke the sampler before its output is admitted.
            # Only the final observation before history advances is a completed action.
            obs=steps[len(steps)-1]
            tops = items[pos]['top_logprobs']
            tied = {int(t['token'][9:]) for t in tops if t['logprob']==max(a['logprob'] for a in tops)}
            require(obs['conditioning'] == key and obs['position'] == pos and obs['action'] == ids[pos]
                    and obs['vocab'] == spec['vocab_size'] and 0 <= obs['top'] < obs['vocab']
                    and obs['maxima'] == len(tied) >= 2 and obs['top'] in tied
                    and obs['raw_action_logprob'] == natural['logprobs'][pos], 'invalid full vocabulary tie observation')
            winners[i,pos] = (obs['top'],tied)
            observations.append(dict(index=i,callbacks_per_rank=len(steps),ranks=len(by_rank),**obs))
    require(winners and len(winners)==sum(map(len,targets.values())), 'unobserved configured history')
    evidence=dict(round_sha256=rr['sha256'], declaration_sha256=pin(packet/'ARGMAX-DELTA.json')['sha256'],
                  container_id=cid, observations=observations, requests=width)
    return winners,evidence
