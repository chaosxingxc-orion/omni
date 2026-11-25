"""Seal S1 membership only from complete real screening and matching prepared TRAIN inputs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from build import pin,save
from prepare_s1_recordings import read,checked,needed
from s1_panel_selection import observed,sample
from source.config_profile import resolve,bind_environment


def member(s,obs,static,out):
    """Join a single real screening result to its recording; offline facts stay separate."""
    rid=s['reference_id'];record=read(checked(static['record_ref']));origin=read(checked(static['recording_origin_ref']))
    packet=read(checked(s['packet']));b=packet['source_binding']
    needed(static['reference_id']==record['recording_ref_id']==origin['recording_ref_id']==b['reference_id']==rid
           and record['online']['node_input_ref']==s['packet'] and packet['branch']==static['fold']==s['fold']
           and record['domain']==origin['domain']==b['domain'] and b['source_component']==s['component']
           and record['split']==origin['split']==b['split']=='TRAIN'
           and record['source_id']==origin['source_id']==b['source_id'] and packet['wire_sha256']==obs['source_wire'],
           'screening observation and static input differ')
    envelope=obs['receipt'];needed(envelope['stage']=='S1' and envelope['request_id']==packet['request_id']
        and envelope['wire_ref']['sha256']==obs['source_wire'],'screening reply belongs to another call')
    observation=dict(schema='node-screen-observation-v1',node='S1',split='TRAIN',domain=b['domain'],reference_id=rid,
        sample_id=record['sample_id'],source_id=b['source_id'],source_component=b['source_component'],
        source_record_sha256=static['source_record_sha256'],packet_ref=s['packet'],source_wire=obs['source_wire'],
        packet_version=packet['version'],fold=s['fold'],supervision=s['supervision'],outcome=obs['outcome'],
        response_ref=envelope['response_ref'],collection_ref=obs['seal'])
    key=static['record_ref']['sha256']
    record=dict(record,screen_observation_ref=save(out/'observations'/(key+'.json'),observation),
                outcome='success' if obs['outcome'] in ('correct_empty','correct_positive') else 'error')
    return dict(record_ref=save(out/'records'/(key+'.json'),record),recording_origin_ref=static['recording_origin_ref'],
                reference_id=rid,fold=s['fold'],outcome=obs['outcome'],panel_conditional_inclusion=s['panel_conditional_inclusion'],
                fold_probability=s['fold_probability'],outcome_conditional_inclusion=s['outcome_conditional_inclusion'])


def seal(cfg):
    out=Path(cfg['output']);evidence=read(checked(cfg['screen']));static=read(checked(cfg['static_inputs']))
    needed(static['schema']=='s1-static-training-inputs-v1' and evidence['schema']=='s1-outcome-screen-v1'
           and static['partition']==evidence['configuration']['partition'],'different S1 panel')
    needed(evidence['panel_complete'] and evidence['observed']==evidence['panel_size']>0,'incomplete frozen S1 observations: %d/%d'%(evidence['observed'],evidence['panel_size']))
    split=read(checked(static['partition']));design=evidence['configuration']
    found,audit=observed(design,split);actual=sample(design,split,found)
    needed(evidence==dict(actual,configuration=design,audit=audit) and actual['selected'],'screen outcomes or sample differs; empty roster')
    rows={s['reference_id']:s for s in static['rows']};reserved={r['component'] for r in split['rows'] if r['split']=='check'}
    draws=[];seen=set()
    for s in actual['selected']:
        rid=s['reference_id'];needed(rid not in seen and rows[rid]['split']==s['split']=='selection' and s['component'] not in reserved,
                                    'duplicate selection or independent-check leakage')
        seen.add(rid);draws.append(member(s,found[rid],rows[rid],out))
    roster=save(out/'ROSTER.json',dict(schema='frozen-s1-screen-roster-v1',split='TRAIN',node='S1',
        inputs=cfg['static_inputs'],screen=cfg['screen'],system_version=design['system_version'],
        execution_version=design['execution_version'],folds=actual['training_folds'],draws=draws,
        cells=dict(Counter(d['fold']+'/'+d['outcome'] for d in draws)),count=len(draws),
        panel_size=actual['panel_size'],frozen_membership=True,fresh_actions=False))

    pool=dict(schema='node-panel-selection-v1',node='S1',split='TRAIN',sealed=True,synthetic_fixture=False,
              population_scope='frozen_recording_fold_panel',population=len(split['rows']),inputs_ref=roster,
              system_version=design['system_version'],execution_version=design['execution_version'],version=roster['sha256'])
    sampling=dict(schema='recording-fold-sampling-v1',purpose='screen',node='S1',scope='node',split='TRAIN',
                  seed=design['seed'],replacement='forbid',format='separated-json-v1',sample_count=len(draws),
                  pool_version=pool['version'],system_version=pool['system_version'])
    save(out/'SAMPLING.json',sampling)
    save(out/'NODE-POOL.json',dict(pool,sampling=sampling))
    return roster


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('config');cli.add_argument('--profile',required=True)
    a=cli.parse_args();profile=read(a.profile)
    if profile['schema']=='omni-local-profile-v2':profile,_=bind_environment(profile,os.environ)
    c=resolve(read(a.config),profile);needed(c['schema']=='s1-screen-roster-config-v1','unexpected roster config')
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'roster':Path(c['output']).parent},min_free_fraction=.1,min_free_bytes=20<<30)
    print(json.dumps(seal(c)))
