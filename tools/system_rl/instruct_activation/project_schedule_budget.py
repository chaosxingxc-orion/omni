"""Isolate one scheduler token budget in a copy of the measured tile engine."""
from pathlib import Path
import sys
from project_prefill_tile import build as tile
from project_screen_lanes import read,save
from decode_lineage import snapshot,changes,digest

def build(spec):
    keys = {'batched_tokens', 'scheduled_tokens'} & spec.keys()
    if len(keys) != 1:
        raise ValueError('select one scheduler control')
    key = keys.pop()
    budget = spec[key]
    flag = {'batched_tokens': '--max-num-batched-tokens',
            'scheduled_tokens': '--max-num-scheduled-tokens'}[key]
    if type(budget)!=int or budget<128 or budget%128:raise ValueError('expected whole KV pages')
    tile(spec)
    root=Path(spec['projection']['target']);parent=Path(spec['measured_parent'])
    before=snapshot(parent)
    if snapshot(root)!=before:raise ValueError('parent runtime differs')
    if spec.get('stage_trace'):
        from project_stage_trace import bind
        bind(spec,root,parent)
        # Compare the CLI change against this exact copy with diagnostic probes.
        before=snapshot(root)
    for lane in spec['projection']['lanes']:
        folder=root/('lane-'+lane);p=folder/'config.json';cfg=read(p)
        if flag in cfg['serve_args'] or '--no-enable-chunked-prefill' in cfg['serve_args']:
            raise ValueError('unexpected parent scheduling options')
        cfg['serve_args'].extend([flag,str(budget)]);save(p,cfg)
        pins=read(folder/'PINS.json');pins[p.name]=digest(p);save(folder/'PINS.json',pins)
    d=changes(before,snapshot(root))
    if d['rows'] or d['files'] or set(d['engine'])!={'serve_args'} or d['engine']['serve_args']['new']!=before['engine']['serve_args']+[flag,str(budget)]:
        raise ValueError('unrelated runtime change')
    save(root/'SCHEDULING.json',dict(control=key,value=budget,parent_sha256=digest(parent/'CANDIDATE.json'),changes=d))

if __name__=='__main__':build(read(Path(sys.argv[1])))
