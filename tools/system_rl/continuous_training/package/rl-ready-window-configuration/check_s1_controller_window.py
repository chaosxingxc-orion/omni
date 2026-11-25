"""Replay a closed collection through the same S1 seal, dataset and window readers, on CPU."""
import argparse,base64,collections,json,sys,tempfile
from pathlib import Path
from build import pin,save
from prepare_s1_recordings import read,checked,needed
from prepare_s1_training_inputs import pcm
from s1_panel_selection import classify,packet_parser,sample,observed
from seal_s1_training_roster import seal
from source.config_profile import digest


def check(history,release,output):
    sys.path.insert(0,str(release/'tree/parents/jev-system-rl/implementation'))
    import dependencies  # Root paths are derived from the actual packaged module.
    sys.path[:0]=[str(Path(__file__).resolve().parent/'source'),str(release/'tree/parents/jev-system-rl/implementation'),
                  str(release/'tree/parents/fresh-cohort-successor')]
    from next_input import bind_screen_sampling
    from node_screen_selection import selection,phase_inputs
    from sampled_training import prepare,bind_window
    from training_references import window_draws
    from fresh_adapter import policy as scope_policy
    import policy
    scope_policy(policy)
    from source import cohort
    from source.fresh_material import node_screen_packet
    h=read(history);coll=read(checked(h['seal']));ver=read(checked(coll['version']));ds=read(checked(h['dataset']))
    meta={d['domain']:[json.loads(line) for line in checked(d['metadata']).open()] for d in ds['domains']}
    rows=[];static=[];found={};parse=packet_parser();components=set();errors=[]
    def rejected(name,fn):
        try:fn()
        except ValueError:errors.append(name)
        else:raise AssertionError(name+' not rejected')
    # One independent recording per component. This is a reader replay, not a
    # new sample, a formal experiment input, or an estimate of population rates.
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        tmp=Path(temp)
        for call in read(checked(ver['receipts'])):
            if call['stage']!='S1' or call['status']!='known':continue
            p=call['packet'];b=p['source_binding'];comp=b['source_component']
            if comp in components:continue
            components.add(comp);rid=b['reference_id'];dom,line,idx=rid.split(':');raw=meta[dom][int(line)-1]
            needed(b['source_id']==dom+':'+str(raw['slurp_id']) and p['sample_id'].endswith(':'+Path(raw['recordings'][int(idx)]['file']).stem),'wrong official recording')
            fold=p['branch'];types=p['allowed_types'];sup=dict(annotated_types=types,present_types=[t for t in types if t in {e['type'] for e in raw['entities']}])
            kind,miss,extra=classify(p,sup,call['receipt'],parse)
            env=call['receipt']['native_envelope'];key=tmp/str(len(rows));packet=save(key/'packet.json',p)
            part,=[a for m in p['wire']['messages'] if isinstance(m['content'],list) for a in m['content'] if a.get('type')=='input_audio']
            origin=dict(recording_ref_id=rid,source_id=b['source_id'],source='SLURP',domain=dom,split='TRAIN',capture_recording_id=rid,
                        recording_filename=raw['recordings'][int(idx)]['file'],audio_ref=pcm(tmp/'media',base64.b64decode(part['input_audio']['data'],validate=True)),context_audio_refs=[])
            binding=dict(node_input_ref=packet);record=dict(recording_ref_id=rid,sample_id=p['sample_id'],source_id=b['source_id'],domain=dom,
                        split='TRAIN',source='SLURP',synthetic_fixture=False,online=binding)
            branch='selection' if rows else 'check'
            rows.append(dict(reference_id=rid,fold=fold,component=comp,split=branch,stratum='/'.join((dom,fold,'present' if sup['present_types'] else 'absent')),
                packet=packet,supervision=sup,panel_conditional_inclusion=1.,fold_probability=1.))
            static.append(dict(reference_id=rid,split=branch,fold=fold,source_record_sha256=digest(raw),record_ref=save(key/'static.json',record),
                               recording_origin_ref=save(key/'origin.json',origin)))
            found[rid]=dict(outcome=kind,missed=miss,extra=extra,receipt=env,seal=h['seal'],source_wire=p['wire_sha256'])
            if len(rows)==10:break
        needed(len(rows)>=4,'insufficient historical components')
        split=dict(schema='s1-panel-partition-v1',rows=rows);sr=save(tmp/'partition.json',split)
        design=dict(schema='s1-outcome-config-v1',partition=sr,collections=[h['seal']],seed=17,training_folds='all',
                    cap_per_domain_fold_outcome=dict(miss=6,extra=6,mixed=6,correct_empty=6,correct_positive=6),
                    system_version=ver['system_version'],execution_version=ver['execution_version'])
        result=sample(design,split,found)
        measured,audit=observed(design,split)
        needed(measured==found,'historical replay and selected receipts differ')
        evidence=dict(result,configuration=design,audit=audit)
        cfg=dict(output=str(tmp/'seal'),screen=save(tmp/'score.json',evidence),static_inputs=save(tmp/'inputs.json',
                 dict(schema='s1-static-training-inputs-v1',partition=sr,rows=static)))
        seal(cfg);pr=pin(tmp/'seal/NODE-POOL.json');pool=read(checked(pr));sampling=pool['sampling'];n=sampling['sample_count']
        response=dict(complete=True,screen_scope='node',node='S1',screen_ref=pr,sampling_config_ref=pin(tmp/'seal/SAMPLING.json'),
            population_scope=pool['population_scope'],population=pool['population'],system_version=pool['system_version'])
        phase=dict(id='historical reader replay',start_step=0,target_nodes=['S1'],rollout_scope='node',frozen_upstream_release=pr['sha256'])
        source=bind_screen_sampling(dict(sampling={},rollout_scope='node',phase_frozen_upstream_ref=pr,
            clock_state_ref=save(tmp/'clock.json',dict(successful_optimizer_steps=0)),learning_config_ref=save(tmp/'learning.json',dict(phases=[phase])),
            policy={},capture={},native=dict(attempt_id='historical-replay'),consumer_config_ref=save(tmp/'consumer.json',{})),response)
        spec=dict(schema='frozen-sampled-training-v1',sample_count=n,queries_per_cohort=3,resample='after_current_dataset_complete',zero_signal='consume_without_optimizer_step')
        conf=dict(execution_mode='formal',sampled_training=spec,max_optimizer_updates=n,root_seed=design['seed'])
        initial=save(tmp/'source.json',source);state=dict(input_ref=initial,screen=response);prepare(conf,state,tmp)
        ds=read(checked(state['sampled_dataset_ref']));selected=read(checked(ds['selected_ref']))
        ids=[];sizes=[]
        for i in range(0,n,3):
            state['sampled_consumed']=i;wr=bind_window(conf,state,initial,tmp/'windows'/str(i));src=read(checked(wr))
            plan=cohort.draw(src,tmp,persist=False); w=read(checked(src['sampled_window_ref']))
            needed(plan['screen_population_scope']==pool['population_scope'] and plan['full_screen_population']==len(rows),'misreported panel')
            exp=window_draws(w,ds,selected); sizes.append(len(exp));mapping=read(checked(read(checked(src['consumer_config_ref']))['recording_map_ref']))
            for d,item,origin in zip(plan['draws'],exp,mapping['records'],strict=True):
                rid=d['record']['recording_ref_id'];dom,line,_=rid.split(':');raw=meta[dom][int(line)-1]
                needed(d['record']==item['record'] and node_screen_packet(d,phase,dict(raw,recording_ref_id=rid,source_record_sha256=digest(raw)),origin)['branch']==d['cell']['fold'],'window input changed')
                ids.append(rid)
        needed(ids==[s['reference_id'] for s in result['selected']] and len(set(ids))==n,'changed membership/order')
        rejected('wrong phase',lambda:phase_inputs(source,dict(phase,target_nodes=['P']),pool))
        rejected('wrong population',lambda:bind_screen_sampling({},dict(response,population=119881)))
        rejected('wrong selection seed',lambda:selection(pool,dict(sampling,seed=sampling['seed']+1)))
        rejected('incomplete panel',lambda:seal(dict(cfg,screen=save(tmp/'partial.json',dict(evidence,observed=0,panel_complete=False)))))
    return save(output,dict(source_collection=h['seal'],screened=len(rows),selected=n,window_sizes=sizes,rejected=errors,
                            production_panel_responses_added=0,model_calls=0,formal_training_set=False))

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('history','release','output'):cli.add_argument(k,type=Path)
    a=cli.parse_args();print(json.dumps(check(a.history,a.release,a.output)))
