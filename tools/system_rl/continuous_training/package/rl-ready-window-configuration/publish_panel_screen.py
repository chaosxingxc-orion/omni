"""Return a sealed, dedicated node-panel DATA result to the existing training controller."""
import argparse,json,os,sys
from pathlib import Path
from build import save
from prepare_s1_recordings import checked,read,needed
from source.config_profile import stage_config,bind_environment

def result(cfg):
    from node_screen_selection import inherited_cost
    request=read(checked(cfg['request_ref']));pool=read(checked(cfg['pool_ref']))
    roster=read(checked(pool['inputs_ref']));source=read(checked(request['input_ref']))
    answer=dict(schema='node-panel-data-result-v1',status='complete',fixture=False,complete=True,
        node=pool['node'],screen_scope='node',population_scope=pool['population_scope'],population=pool['population'],
        request_ref=cfg['request_ref'],screen_ref=cfg['pool_ref'],sampling_config_ref=cfg['sampling_ref'],
        usage_ref=roster['usage_ref'],system_version=pool['system_version'],transport_config_ref=source['capture']['transport_config_ref'])
    total=inherited_cost(answer,cfg['request_ref'],request)
    needed(total<=request['budget']['tokens_upper'],'measured panel tokens exceed the original envelope')
    return save(Path(cfg['output']),answer)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--profile',required=True,type=Path)
    a=p.parse_args();site=read(a.profile)
    if site['schema']=='omni-local-profile-v2':site,_=bind_environment(site,os.environ)
    c=stage_config(read(a.config),site,'node-panel-data-return-v1');release=Path(c['release'])
    sys.path.insert(0,str(release/'tree/parents/jev-system-rl/implementation'))
    print(json.dumps(result(c)))
