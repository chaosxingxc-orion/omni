"""Audit explicit runtime members; vary the operator config and read actual consumer outputs."""
import argparse,ast,copy,json
from pathlib import Path
from build import save
from prepare_s1_recordings import read,needed
from source.config_profile import stage_config
from paired_experiment import check_same_action,graph,rank_profile,validate

HERE=Path(__file__).resolve().parent


def audit(manifest,panel_profile,experiment,out):
    rules=read(manifest);scanned=[]
    from release_source import source_path
    files={}
    for member,forbidden in rules['runtime_members'].items():
        p=(HERE/member).resolve()
        needed(p.is_relative_to(HERE) and p.is_file(),'unexpected audit member')
        files[member]=(p,forbidden)
    for member,forbidden in rules['transition_members'].items():
        name='@transition/'+member
        files[name]=(source_path(HERE.parent,name),forbidden)
    for member,(p,forbidden) in files.items():
        vals={n.value for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Constant) and type(n.value) in (float,int,str)}
        needed(not vals.intersection(forbidden),'experiment literal reintroduced: '+member)
        scanned.append(member)
    from sample_s1_folds import select
    cfg=read(HERE/'S1-EXPERIMENT.template.json');stage='s1-fold-panel-v1'
    def render(c):return select(stage_config(c,panel_profile,stage))
    original=render(cfg);variant=copy.deepcopy(cfg)
    variant['parameters'].update(SEED=31,PER_FOLD_DOMAIN={'present':3,'absent':2},FOLDS=[original['draws'][0]['fold']])
    v=render(variant);needed(len(v['draws'])==10 and len(original['draws'])>10
        and all(x['fold']==variant['parameters']['FOLDS'][0] for x in v['draws']),'fold quota did not reach the actual sampler')
    v2=copy.deepcopy(variant);v2['parameters']['SEED']+=1
    needed(render(v2)['draws']!=v['draws'],'seed ignored')
    launch=stage_config(cfg,dict(schema='omni-local-profile-v1',values=dict(INITIAL_SOURCE_REF='source',TRAINING_INPUT_REF='input')),'formal-target-start-v2')
    needed(launch['target']=='S1' and launch['initialization']['source_ref']=='source' and launch['training_input_ref']=='input','launch stage did not consume its profile')
    a=experiment;b=dict(a,lora=dict(a['lora'],lora_rank=128),actor_batch=dict(a['actor_batch'],gradient_accumulation_steps=1),
        episode_concurrency=64,graph_widths=[1,4,8,16,32,64])
    needs=validate(b);needed(rank_profile(b)['inference']['max_lora_rank']==b['lora']['lora_rank'] and graph(b)['max_num_seqs']==64,'actor/service projection differs')
    verdict=dict(policy='test',formal_updates=0,tokens=2,rows=1,nonzero=1,max_abs=1e-8)
    tol={'same_action_max_abs':1e-8};check_same_action(verdict,'test',tol);bad=0
    for obj,settings in [(dict(verdict,max_abs=1.0001e-8),tol),(dict(verdict,max_abs=float('nan')),tol),(verdict,{})]:
        try:check_same_action(obj,'test',settings)
        except ValueError:bad+=1
        else:raise AssertionError('invalid precision acceptance')
    return save(out,dict(scanned=scanned,panel_original=len(original['draws']),panel_variant=len(v['draws']),
        selected_fold=variant['parameters']['FOLDS'][0],seed_changes_members=True,actor_variant_recordings=needs,
        inference_variant_max_seqs=graph(b)['max_num_seqs'],actor_variant_rank=b['lora']['lora_rank'],precision_negative_cases=bad,
        model_calls=0,scope='configured sources and CPU consumers only'))

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ('rules','panel_profile','experiment','output'):cli.add_argument(n,type=Path)
    a=cli.parse_args();print(json.dumps(audit(a.rules,read(a.panel_profile),read(a.experiment),a.output)))
