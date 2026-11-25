"""Select a staged actor reader after a complete predecessor, inside the iteration guard."""
from pathlib import Path


def install(config,spec,base):
    import formal_actor_transport as transport
    from dependencies import require
    from run_package import read,pin,write
    new=read(spec['actor_reader_release_ref'])
    old=config['formal_actor']
    require(new['parent_template']==old['stage_template_ref'] and new['iteration_ref']==spec['iteration_ref'],
            'actor revision targets different experiment')
    template=read(new['stage_template']); lock=read(new['source_lock'])
    require(lock['precision_lock_sha256']==old['continuous_actor']['precision_lock_sha256'],'precision changed')
    original=transport.train;end_original=transport.end_resident
    oldclient=Path(config['output_root'])/'formal-resident-client'
    switch=base/'ACTOR-SWITCH.json'
    def apply(cfg):
        transition=read(switch);actor=cfg['formal_actor']
        cfg['_restored_command']=transition['command'];cfg['_replacement_checkpoint']=transition['checkpoint']
        cfg['_resident_client_root']=str(base/'actor-client')
        actor['stage_template_ref']=transition['restore_template']
        actor['continuous_actor']=dict(actor['continuous_actor'],source_lock_ref=new['source_lock'])
    def train(cfg,state,directory,guard):
        if not switch.exists() and read(state['capture']['input_ref'])['capture'].get('training_input_layout')=='group_refs_v1':
            require(not (directory/'SUBMIT-INTENT.json').exists() and not (directory/'RESIDENT-CALL.json').exists(),
                    'unreconciled train intent at replacement')
            records=sorted((oldclient/'completed').glob('*.json'));command=read(records[-1]);cp=state['resume_ref']
            require(command['owner']==read(oldclient/'OWNER.json')['target'] and
                    command['sequence']==len(records)-1 and command['checkpoint_sha256']==cp['sha256'] and
                    command['source_boot']==old['source_boot'],'different terminal actor command')
            area=Path(command['owner'])/'hot/resident/results'/format(command['sequence'],'06d')
            require(Path(cp['path']).parents[1]==area and read(cp)['step']==state['step'],'not the exact update boundary')
            import copy
            stage=copy.deepcopy(template);m=stage['capture_mounts']
            m['logical_roots']['closed_actor_command']=str(area)
            m['formal_input_roles'].append('closed_actor_command')
            restore=write(base/'RESTORE-ACTOR.json',stage)
            ended=end_original(cfg,guard)
            require(ended and read(ended)['checkpoint_ref']['sha256']==cp['sha256'],'old actor closed another checkpoint')
            write(switch,dict(command=command,checkpoint=cp,restore_template=restore,ended=ended))
        if switch.exists():apply(cfg)
        return original(cfg,state,directory,guard)
    transport.train=train
    if switch.exists():apply(config)
