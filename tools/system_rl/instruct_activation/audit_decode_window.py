"""Read a sealed paired text-decode window and check the request-weighted rate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from io_utils import canonical, doc, pin, require
from measure_existing_services import work_metrics


def audit(root, reference_ms, factor):
    require(math.isfinite(reference_ms) and reference_ms > 0 and math.isfinite(factor) and factor >= 1,
            'positive finite time reference and factor required')
    started = doc(pin(root/'STARTED.json'))
    config = doc(started['config_ref'])
    final_ref = pin(root/'RESULT.json')
    final = doc(final_ref)
    drain_ref = pin(root/'DRAINED.json')
    require(doc(drain_ref)['guards_releasable'] and not list(root.rglob('UNKNOWN.json')) and
            final['passed'] and final['precision_acceptance'] is False and
            final['training_use'] is False and not config.get('decode_replay') and
            Path(config['output']).resolve() == root.resolve(), 'expected a closed, probe-free performance window')
    require(config['round_order'] in (['paired'], ['paired','paired_warm']) and
            {s['name'] for s in config['services']} == {'baseline','candidate'}, 'paired measurement required')
    expected = [(b,n,arm) for b in range(len(config['grouped_blocks']))
                for n in config['grouped_concurrency'] for arm in config['round_order']]
    require(len(expected) == len(final['rounds']) > 0, 'missing rounds')
    reports = []
    for i, ((b,n,arm), ref) in enumerate(zip(expected,final['rounds'])):
        r = doc(ref)
        sub = root/f'round-{i}'
        require(Path(ref['path']) == sub/'RESULT.json' and (r['block'],r['concurrency_per_service'],r['arm']) ==
                (b,n,arm) and r['input_refs'] == config['grouped_blocks'][b], 'round identity mismatch')
        salt = hashlib.sha256(canonical([config['measurement_id'],b,'paired',n])).hexdigest()
        wires = {(g,j):doc(w) for g,ws in enumerate(r['input_refs']) for j,w in enumerate(ws)}
        rows = {}
        for lane in ('baseline','candidate'):
            responses = {}
            prompts = tokens = 0
            for row in r['requests']:
                if row['service'] != lane:
                    continue
                key = row['group'],row['replicate']
                wire = wires[key]; loc = sub/lane/f'{key[0]}-{key[1]}'
                require(key not in responses and Path(row['receipt_ref']['path']) == loc/'RESPONSE.json'
                        and wire['top_logprobs'] == 0 and not wire.get('vllm_xargs')
                        and wire['return_token_ids'] and wire['logprobs'] and not wire['stream'],
                        'probe or duplicate request')
                intent = doc(pin(loc/'INTENT.json'))
                require(intent['wire_sha256'] == hashlib.sha256(canonical(dict(wire,cache_salt=salt))).hexdigest(),
                        'request or cache namespace changed')
                receipt = doc(row['receipt_ref'])
                require(receipt['status'] == 200 and hashlib.sha256(receipt['raw'].encode()).hexdigest()
                        == receipt['response_sha256'], 'invalid HTTP receipt')
                body = json.loads(receipt['raw']); choice, = body['choices']; ids=choice['token_ids']
                content=choice['logprobs']['content']; usage=body['usage']
                require(usage == row['usage'] and len(ids) == len(content) == usage['completion_tokens'] > 1
                        and all(item['token'] == f'token_id:{tid}' and math.isfinite(item['logprob'])
                                for tid,item in zip(ids,content)), 'invalid score or denominator')
                responses[key] = ids,[x['logprob'] for x in content]
                prompts += usage['prompt_tokens']; tokens += len(ids)
            require(responses.keys() == wires.keys(), 'missing HTTP result')
            before,after=(doc(pin(sub/lane/(k+'.json')))['metrics'] for k in ('BEFORE','AFTER'))
            m = work_metrics(before,after)
            require(m['prefix_queries'] == prompts and 0 <= m['prefix_hits'] <= prompts and
                    m['tpot_requests'] == len(wires) and m['itl_intervals'] == tokens-len(wires) and
                    m['mean_request_tpot_seconds'] > 0 and m['mean_itl_seconds'] > 0 and
                    all(v == m[k] for k,v in r['metrics'][lane].items()), 'unexpected engine counters')
            rows[lane] = responses,m
        left,right = [rows[x][0] for x in ('baseline','candidate')]
        same = [key for key in wires if left[key][0] == right[key][0]]
        m = rows['candidate'][1]; ms = 1000*m['mean_request_tpot_seconds']
        reports.append(dict(round_sha256=ref['sha256'],block=b,concurrency=n,arm=arm,requests=len(wires),
                            intervals=int(m['itl_intervals']),request_tpot_ms=ms,itl_ms=1000*m['mean_itl_seconds'],
                            factor=ms/reference_ms,within_target=ms<=reference_ms*factor,
                            matched=len(same),same_token_and_score=sum(left[k]==right[k] for k in same),
                            matched_actions=sum(len(left[k][0]) for k in same)))
    candidate, = [s for s in config['services'] if s['name']=='candidate']
    return dict(result_sha256=final_ref['sha256'],drained_sha256=drain_ref['sha256'],
                candidate_container=candidate['container'], source_boot=config['source_boot'],
                run_config_sha256=started['config_ref']['sha256'],reference_ms=reference_ms,
                limit_factor=factor,rounds=reports,all_rounds_within_target=all(r['within_target'] for r in reports),
                precision_acceptance=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds',required=True,type=Path); p.add_argument('--reference-ms',required=True,type=float)
    p.add_argument('--factor',required=True,type=float)
    args = p.parse_args()
    print(json.dumps(audit(args.rounds.resolve(),args.reference_ms,args.factor),indent=2,allow_nan=False))
