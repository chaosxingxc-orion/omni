"""Prepare immutable P prompt and annotation references before policy activation."""
import json
import sys
from pathlib import Path


def prepare(config_path, lengths_path, out):
    from cohort_io import read, pin, save, digest, require
    from official import sources, annotation
    from material import node_screen_packet
    from execution import measured_prompt_index, PromptLengths
    cfg=read(pin(config_path))
    window=read(cfg['sampled_window_ref'])
    dataset=read(window['dataset_ref'])
    profile=cfg['capture']['profile_ref']
    lengths_ref=pin(lengths_path)
    lengths=measured_prompt_index(lengths_ref,cfg['native']['model_path'])
    idx=read(lengths_ref)
    require(dataset['statistics']['inputs_ref']==idx['inputs_ref'] and idx['profile_ref']==profile,
            'lengths from different TRAIN inputs or sampler')
    pop=read(idx['inputs_ref'])
    require(pop['formal_sample_locked'] and len(pop['draws'])==dataset['count']==pop['rows'],
            'incomplete frozen TRAIN inputs')
    rows={}
    with sources(cfg) as (official,_):
        for draw in pop['draws']:
            r=read(draw['record_ref']); origin=read(draw['recording_origin_ref']); key=digest(r)
            label=annotation({'record':r},official)
            packet=node_screen_packet({'record':r},{'rollout_scope':'node','target_nodes':['P']},label,origin)
            # Keep the original per-record packet by reference; never store sampled actions.
            rid=r['recording_ref_id']
            entry=dict(record=key,origin=digest(origin),packet_ref=r['online']['node_input_ref'],annotation=label)
            require(rid not in rows or rows[rid]==entry,'conflicting immutable recording')
            rows[rid]=entry
    return save(out,dict(schema='p-static-supply-v1',dataset_ref=window['dataset_ref'],source_sha256=cfg['capture']['official_source_config_ref']['sha256'],
        profile_ref=profile,length_index_ref=lengths_ref,rows=rows))


def for_window(cfg,plan):
    from cohort_io import read, digest, require
    ref=cfg['capture'].get('static_supply_ref')
    if not ref:return None
    window=read(plan['sampled_window_ref'])
    macro=cfg['capture'].get('recording_macros_ref')
    entries=None
    if macro:
        pages=read(macro);width=pages['width'];start=window['start'];end=window['end']
        require(pages['static_supply_ref']==ref and window['dataset_ref']==pages['dataset_ref'] and
                0<=start<end<=pages['count'] and end-start==len(plan['draws']),'macro/window mismatch')
        entries=[]
        for i in range(start//width,(end-1)//width+1):
            page=read(pages['pages'][i]);offset=i*width
            require(page['start']==offset,'wrong static page')
            entries.extend(page['rows'][max(0,start-offset):min(width,end-offset)])
        require(len(entries)==len(plan['draws']),'short static page')
    index=read(ref) if entries is None else pages['header']
    require(index['schema']=='p-static-supply-v1' and plan['sampling']['node']=='P' and
            read(plan['sampled_window_ref'])['dataset_ref']==index['dataset_ref'] and
            cfg['capture']['official_source_config_ref']['sha256']==index['source_sha256'] and
            cfg['capture']['profile_ref']==index['profile_ref'] and
            cfg['capture']['prompt_length_index_ref']==index['length_index_ref'],'static supply version differs')
    rows=[]
    for i,draw in enumerate(plan['draws']):
        r=draw['record']
        if entries is not None:
            require(draw['dataset_draw_index']==window['start']+i and entries[i]['id']==r['recording_ref_id'],'macro recording order changed')
        entry=index['rows'][r['recording_ref_id']] if entries is None else entries[i]['entry']
        require(entry['record']==digest(r) and entry['packet_ref']==r['online']['node_input_ref']
                and entry['origin']==digest(read(draw['recording_origin_ref'])),'changed rollout material')
        rows.append(entry)
    return rows


if __name__=='__main__':
    pars,cfg,lengths,out=map(Path,sys.argv[1:])
    sys.path.insert(0,str(pars/'fresh-cohort-successor'))
    print(json.dumps(prepare(cfg,lengths,out)))
