"""Compare real HF ranks with a closed TP4 replay; never grant admission."""
import argparse
import json
import math
from pathlib import Path

from collect_retained_boundaries import store
from compare_retained_score_paths import metrics


def read(p):
    return json.loads(p.read_bytes())


def hf_reference(hf, world):
    if world not in (4, 8) or sorted(r['rank'] for r in hf) != list(range(world)):
        raise ValueError(f'{world} distinct HF ranks required')
    rank0 = min(hf, key=lambda r: r['rank'])
    keys = ('world_size', 'input_pins', 'asset_sha256', 'internal_format', 'full_vocabulary',
            'argmax', 'scores', 'predictor_position', 'rows', 'updates',
            'gate_candidate', 'attention_candidate', 'down_candidate')
    if any(row['world_size'] != world or row['updates'] != 0 or
           any(row.get(k) != rank0.get(k) for k in keys) for row in hf):
        raise ValueError('HF ranks disagree on the same original input')
    return rank0


def compare_hf_topology(hf4, hf8):
    a, b = hf_reference(hf4, 4), hf_reference(hf8, 8)
    for key in ('input_pins','asset_sha256','hardware','internal_format','predictor_position',
                'gate_candidate','attention_candidate','down_candidate'):
        if a.get(key) != b.get(key):
            raise ValueError('topology comparison changed '+key)
    if (not a['scores'] or len(a['scores']) != len(b['scores'])
            or len(a['argmax']) != len(a['scores']) or len(b['argmax']) != len(b['scores'])
            or list(a['rows']) != list(b['rows'])):
        raise ValueError('incomplete paired topology observations')
    changed = [k for k,v in a['rows'].items() if v != b['rows'][k]]
    return dict(hf4_vs_hf8=metrics(a['scores'],b['scores'],.01), changed_boundaries=changed,
                observed_boundaries=len(a['rows']), first=next(iter(changed),None),
                raw_top1_flips=sum(x!=y for x,y in zip(a['argmax'],b['argmax'],strict=True)),
                elapsed_seconds={str(n):[r['elapsed_seconds'] for r in arm]
                                 for n,arm in ((4,hf4),(8,hf8))}, production_acceptance=False)


def compare_actor_vjp(reference, live, old_spec, new_spec):
    """Paired fixed-action diagnostic; permit only the explicitly pinned VJP additions."""
    old, new = hf_reference(reference, 4), hf_reference(live, 4)
    ignore = {'name', 'hot', 'source_pins', 'files', 'actor_vjp'}
    if ({k:v for k,v in old_spec.items() if k not in ignore} !=
            {k:v for k,v in new_spec.items() if k not in ignore}
            or old_spec.get('actor_vjp') or new_spec.get('actor_vjp') is not True):
        raise ValueError('changed model, topology or execution configuration')
    for key, exceptions in [('source_pins', {'retained_hf_fsdp_control.py'}),
                            ('files', {'retained_attention_candidate.py','retained_expert_candidate.py'})]:
        a,b=old_spec[key],new_spec[key]
        if set(a)!=set(b) or any(a[k]!=b[k] for k in a.keys()-exceptions):
            raise ValueError('changed sealed inputs or worker source')
    if old['input_pins']!=old_spec['files'] or new['input_pins']!=new_spec['files']:
        raise ValueError('result pins do not match execution configuration')
    for k in ('hardware','asset_sha256','internal_format','predictor_position'):
        if old[k]!=new[k]:
            raise ValueError('different reference: '+k)
    g=new['actor_gradient']
    if (not g or any(r['actor_gradient']!=g for r in live) or g['optimizer_steps']!=0
            or g['globally_covered']!=6684672 or not math.isfinite(g['global_norm'])
            or g['global_norm']<=0 or not math.isfinite(g['loss'])):
        raise ValueError('incomplete actor gradient evidence')
    for a in (old['scores'],new['scores'],g['live_scores']):
        if (not a or len(a)!=len(old['argmax']) or not all(map(math.isfinite,a))):
            raise ValueError('incomplete finite action scores')
    if len(old['rows'])!=98 or old['rows'].keys()!=new['rows'].keys():
        raise ValueError('incomplete paired boundaries')
    for k in ('gate_candidate','down_candidate','attention_candidate'):
        if (old[k]['arithmetic']!=new[k]['arithmetic'] or set(new[k]['layers'])!={str(i) for i in range(48)}
                or set(old[k]['layers'])!=set(new[k]['layers']) or not all(v>0 for v in new[k]['layers'].values())):
            raise ValueError('candidate scope changed: '+k)
    changed=[k for k,v in old['rows'].items() if new['rows'][k]!=v]
    frozen=metrics(old['scores'],new['scores'],.01)
    train=metrics(old['scores'],g['live_scores'],.01)
    return dict(reference_vs_frozen=frozen,reference_vs_grad_enabled=train,
                frozen_logits_digest_equal=old['full_vocabulary']==new['full_vocabulary'],
                frozen_raw_top1_flips=sum(x!=y for x,y in zip(old['argmax'],new['argmax'],strict=True)),
                frozen_observed_boundaries=len(old['rows']),frozen_changed_boundaries=changed,
                gradient_coverage=g['globally_covered'],gradient_norm=g['global_norm'],
                fixed_action_nll=g['loss'],optimizer_steps=0,production_acceptance=False)


def compare(hf, vl, lease):
    rank0 = hf_reference(hf, 4)
    if vl['actual_tp'] != 4 or vl['production_acceptance'] or not lease['score_retained_prefill']:
        raise ValueError('closed diagnostic TP4 replay required')
    for name in ('retained-actions.json', 'P-BEHAVIOR.json'):
        if rank0['input_pins'][name] != lease['input_files'][name]:
            raise ValueError(f'not the same sealed actions: {name}')
    receipts = vl['decode_row_receipts']
    if sorted(r['rank'] for r in receipts) != list(range(4)):
        raise ValueError('missing TP4 execution receipt')
    for key, filename in (('gate_candidate','retained_expert_candidate.py'),
                          ('down_candidate','retained_expert_candidate.py'),
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
    if lease.get('native_audio') is True and (
            control or any('intervention' in r for r in receipts)):
        raise ValueError('native audio comparison requires no audio intervention')
    audio_layers = {str(i) for i in lease.get('audio_attention_layers', ())}
    for row in receipts:
        audio = row.get('audio_attention')
        if bool(audio) != bool(audio_layers) or (audio and (
                audio['arithmetic'] != 'hf_full_head_windowed_bf16_sdpa'
                or set(audio['layers']) != audio_layers
                or any(type(x['calls']) is not int or x['calls'] <= 0
                       for x in audio['layers'].values()))):
            raise ValueError('live audio attention candidate execution differs')
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
    trace_expected = lease.get('trace_predictor', True)
    if trace_expected is False and lease.get('native_audio') is not True:
        raise ValueError('untraced comparison is reserved for native audio replay')
    for row in p.get('paired_trace', ()):
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
    if sorted(r['rank'] for r in layers) != (list(range(4)) if trace_expected else []):
        raise ValueError('incomplete TP4 rank evidence')
    def top1(values):
        if not values or any(type(v) is not int or not 0<=v<rank0['full_vocabulary']['shape'][-1] for v in values):
            return None
        return dict(compared=len(values), mismatches=sum(x!=y for x,y in zip(rank0['argmax'],values)))
    prefill_top1 = top1(p['forced_prefill'].get('observed_raw_top1'))
    decode_top1 = top1(p.get('matching_prefix_raw_top1'))
    extra = {}
    if lease.get('prefix_cache_pair') == 'P':
        w = p['cache_pair']
        scores = w['scores']
        if (w['cold_cached_tokens'] != 0 or type(w['warm_cached_tokens']) is not int
                or not 0 < w['warm_cached_tokens'] <= w['prompt_tokens']
                or len(w['raw_top1']) != len(scores) or len(scores) > len(ref)):
            raise ValueError('cold miss and warm hit required on the same retained prompt')
        extra['cache_pair'] = dict(cold_tokens=0, warm_tokens=w['warm_cached_tokens'],
                prompt_tokens=w['prompt_tokens'], warm_complete=len(scores)==len(ref),
                warm_vs_hf=metrics(ref[:len(scores)],scores,.01),warm_raw_top1=top1(w['raw_top1']))
    return dict(actions=len(ref), predictor_position=rank0['predictor_position'],
                hf_scores=ref, vl_prefill_scores=pref, vl_decode_matching_prefix_scores=decode,
                hf_vs_vl_prefill=metrics(ref,pref,.01), hf_vs_vl_decode=metrics(ref[:len(decode)],decode,.01),
                vl_prefill_vs_decode=metrics(pref[:len(decode)],decode,.01), split_layers=layers,
                controlled_vl_audio=bool(control), decode_complete=len(decode)==len(ref),
                vocabulary=rank0['full_vocabulary']['shape'][-1],
                hf_argmax_observed=True, vl_argmax_compared=bool(prefill_top1 and decode_top1),
                hf_vs_vl_prefill_top1=prefill_top1, hf_vs_vl_decode_top1=decode_top1,
                **extra, production_acceptance=False)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hf',type=Path,required=True)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--vl',type=Path)
    modes.add_argument('--hf8',type=Path)
    modes.add_argument('--actor-vjp',type=Path)
    p.add_argument('--reference-spec',type=Path)
    p.add_argument('--actor-spec',type=Path)
    p.add_argument('--lease',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    h = [read(a.hf/f'rank-{i}'/'RESULT.json') for i in range(4)]
    if bool(a.lease) != bool(a.vl):
        p.error('--lease is required with --vl only')
    if a.actor_vjp:
        if not (a.reference_spec and a.actor_spec):
            p.error('--actor-vjp requires both immutable run specifications')
        result=compare_actor_vjp(h,[read(a.actor_vjp/f'rank-{i}'/'RESULT.json') for i in range(4)],
                                 read(a.reference_spec),read(a.actor_spec))
    else:
        result = (compare_hf_topology(h, [read(a.hf8/f'rank-{i}'/'RESULT.json') for i in range(8)])
              if a.hf8 else compare(h,read(a.vl),read(a.lease)))
    store(a.output,json.dumps(result,indent=2).encode())
