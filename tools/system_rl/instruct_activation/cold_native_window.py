"""Compare separate cold-audio windows using the actual on-wire batches and cache counters."""
import json
from pathlib import Path
from batch_window import analyze
from io_utils import doc, pin, require


def measure(config, lane, adapter, native, gold, parser):
    cfg = doc(pin(config))
    root = Path(cfg['output'])
    window, = [w for w in analyze(config, gold, parser) if w['lane'] == lane]
    run = doc(doc(pin(root/'RESULT.json'))['rounds'][0])
    service, = [v for v in cfg['services'] if v['name'] == lane]
    require((service.get('stock_adapter_alias') == adapter if native else 'stock_adapter_alias' not in service),
            'unexpected serving path')
    width, = cfg['grouped_concurrency']
    require(run['concurrency_per_service'][lane] == width and run['admission'] == 'group_drained', 'different admission')
    warm = root.parents[1]/(cfg['measurement_id'].removesuffix('-latency')+'-warm')
    wc = doc(pin(warm/'CONFIG.local.json'))
    require(doc(pin(warm/'rounds/DRAINED.json'))['guards_releasable']
            and doc(pin(warm/'rounds/RESULT.json'))['passed']
            and wc['start_receipts'][lane] == cfg['start_receipts'][lane], 'unbound warm service')
    wr, = doc(pin(warm/'rounds/RESULT.json'))['rounds']
    warmrun = doc(wr)
    warm_wires = [x['sha256'] for x in warmrun['input_refs']]
    wires = [x['sha256'] for x in run['input_refs']]
    require(not set(wires).intersection(warm_wires) and len(set(wires+warm_wires)) == len(wires+warm_wires)
            and warmrun['concurrency_per_service'][lane] == width and warmrun['admission'] == 'group_drained',
            'warm panel is not independent')
    rows = sorted((r for r in run['requests'] if r['service']==lane), key=lambda r:(r['group'],r['replicate']))
    replies = [json.loads(doc(r['receipt_ref'])['raw']) for r in rows]
    require(all(x['model'] == adapter for x in replies) if native else
            all(x['model'] == doc(w)['model'] for x,w in zip(replies,run['input_refs'],strict=True)), 'wrong response model')
    m = run['metrics'][lane]
    require(m['mm_hits']==0 and m['mm_queries']==len(rows) and window['output_budget']['reason']==['stop'],
            'requires cold audio and natural stop')
    return (warm_wires,wires,[x['prompt_token_ids'] for x in replies],width,m['prefix_hits'],m['prefix_queries']), dict(
        lane=lane, round_sha256=window['round_sha256'],requests=len(rows),valid=window['valid'],
        batch_s=[x['release_to_last_s'] for x in window['batches']], mean_batch_s=window['mean_batch_s'],
        batch_stages=[dict(first_response_s=b['http_finish_from_release_s'][0],
                           first_to_last_s=b['first_to_last_finish_s'],last_only_s=b['last_only_s'],
                           send_spread_s=b['send_spread_s'],output_tokens=b['generated'])
                      for b in window['batches']],
        **{k:m[k] for k in ('mm_hits','mm_queries','prefix_hits','prefix_queries','mean_queue_seconds',
                             'mean_prefill_seconds','mean_ttft_seconds','mean_decode_seconds')})


def compare(cfg):
    native = Path(cfg['cold_native_config'])
    candidate = Path(cfg['candidate_config'])
    def adapter(f):
        c=doc(pin(f))
        return doc(pin(Path(c['output']).parents[1]/(c['measurement_id'].removesuffix('-latency')+'-run.local.json')))['adapter_alias']
    alias = adapter(native)
    require(alias == adapter(candidate), 'different policy')
    g,p = Path(cfg['gold']),Path(cfg['parser'])
    a, left = measure(candidate,cfg['lane'],alias,False,g,p)
    b, right = measure(native,cfg['native_lane'],alias,True,g,p)
    require(a==b, 'different warm wires, measurement inputs, expanded prompts, width or prefix cache')
    return dict(native=right,candidate=left,gap_s=left['mean_batch_s']-right['mean_batch_s'],
                separate_windows=left['round_sha256']!=right['round_sha256'])
