"""Compare real HF ranks with a closed TP4 replay; never grant admission."""
import argparse
import json
from pathlib import Path

from collect_retained_boundaries import store
from compare_retained_score_paths import metrics


def read(p):
    return json.loads(p.read_bytes())


def compare(hf, vl, lease):
    if len(hf) != 4 or sorted(r['rank'] for r in hf) != list(range(4)):
        raise ValueError('four HF ranks required')
    rank0 = hf[0]
    keys = ('world_size', 'input_pins', 'asset_sha256', 'internal_format', 'full_vocabulary',
            'argmax', 'scores', 'predictor_position', 'rows', 'updates',
            'gate_candidate', 'attention_candidate')
    if any(row['world_size'] != 4 or row['updates'] != 0 or
           any(row.get(k) != rank0.get(k) for k in keys) for row in hf):
        raise ValueError('HF ranks disagree on the same original input')
    if vl['actual_tp'] != 4 or vl['production_acceptance'] or not lease['score_retained_prefill']:
        raise ValueError('closed diagnostic TP4 replay required')
    for name in ('retained-actions.json', 'P-BEHAVIOR.json'):
        if rank0['input_pins'][name] != lease['input_files'][name]:
            raise ValueError(f'not the same sealed actions: {name}')
    receipts = vl['decode_row_receipts']
    if sorted(r['rank'] for r in receipts) != list(range(4)):
        raise ValueError('missing TP4 execution receipt')
    for key, filename in (('gate_candidate','retained_expert_candidate.py'),
                          ('attention_candidate','retained_attention_candidate.py')):
        candidate = rank0.get(key)
        if any(bool(r.get(key)) != bool(candidate) for r in receipts):
            raise ValueError('HF and vLLM candidate activation differs: '+key)
        if candidate and (rank0['input_pins'][filename] != lease['input_files'][filename]
                or any(r[key]['arithmetic'] != candidate['arithmetic']
                       or r[key]['layers'].keys() != candidate['layers'].keys()
                       or not all(type(v) is int and v > 0 for v in r[key]['layers'].values())
                       for r in receipts)):
            raise ValueError('HF and vLLM candidate identity differs: '+key)
    control = lease.get('audio_control')
    if control:
        receipts = vl['decode_row_receipts']
        if sorted(r['rank'] for r in receipts) != list(range(4)) or any(
            not r['intervention']['observations'] or
            r['intervention']['reference'] != lease['input_files'][control['file']]
            for r in receipts
        ):
            raise ValueError('audio intervention not observed on every rank')
    p, = (x for x in vl['retained_decode'] if x['node'] == 'P')
    ref, pref, decode = rank0['scores'], p['forced_prefill']['scores'], p['matching_prefix_scores']
    if len(ref) != len(pref) or len(ref) != p['original_tokens'] or len(decode) != p['same_action_prefix']:
        raise ValueError('same action prefix must be explicit')
    layers = []
    for row in p['paired_trace']:
        if row['predictor_position'] != rank0['predictor_position']:
            raise ValueError('predictor positions differ')
        v = row['boundaries']
        if set(v) != rank0['rows'].keys():
            raise ValueError('incomplete common boundaries')
        layers.append(dict(rank=row['rank'],
            hf_prefill_first=next((k for k,x in rank0['rows'].items() if x!=v[k]['prefill_sha256']), None),
            hf_decode_first=next((k for k,x in rank0['rows'].items() if x!=v[k]['decode_sha256']), None),
            vl_prefill_decode_first=row['first_observed_difference'],
            live_vl_attention_operands=row.get('attention_operands')))
    if sorted(r['rank'] for r in layers) != list(range(4)):
        raise ValueError('incomplete TP4 rank evidence')
    def top1(values):
        if not values or any(type(v) is not int or not 0<=v<rank0['full_vocabulary']['shape'][-1] for v in values):
            return None
        return dict(compared=len(values), mismatches=sum(x!=y for x,y in zip(rank0['argmax'],values)))
    prefill_top1 = top1(p['forced_prefill'].get('observed_raw_top1'))
    decode_top1 = top1(p.get('matching_prefix_raw_top1'))
    return dict(actions=len(ref), predictor_position=rank0['predictor_position'],
                hf_scores=ref, vl_prefill_scores=pref, vl_decode_matching_prefix_scores=decode,
                hf_vs_vl_prefill=metrics(ref,pref,.01), hf_vs_vl_decode=metrics(ref[:len(decode)],decode,.01),
                vl_prefill_vs_decode=metrics(pref[:len(decode)],decode,.01), split_layers=layers,
                controlled_vl_audio=bool(control), decode_complete=len(decode)==len(ref),
                vocabulary=rank0['full_vocabulary']['shape'][-1],
                hf_argmax_observed=True, vl_argmax_compared=bool(prefill_top1 and decode_top1),
                hf_vs_vl_prefill_top1=prefill_top1, hf_vs_vl_decode_top1=decode_top1,
                production_acceptance=False)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hf',type=Path,required=True)
    p.add_argument('--vl',type=Path,required=True)
    p.add_argument('--lease',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    h = [read(a.hf/f'rank-{i}'/'RESULT.json') for i in range(4)]
    store(a.output,json.dumps(compare(h,read(a.vl),read(a.lease)),indent=2).encode())
