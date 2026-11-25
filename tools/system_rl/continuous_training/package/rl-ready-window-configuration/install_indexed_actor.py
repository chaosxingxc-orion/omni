"""Select a staged actor reader after a complete predecessor, inside the iteration guard."""
from pathlib import Path


def verify_launcher(release, *, read, pin):
    """Verify every sibling import once at installation, before retiring the current owner."""
    launcher=release['launcher_ref']
    root=Path(launcher['path']).parent
    refs=read(release['launcher_members'])
    files={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if (refs.get('stage_bounded_actor_step.py')!=launcher or files.keys()!=refs.keys()
            or any(path.resolve()!=path or pin(path)!=refs[name] for name,path in files.items())):
        raise ValueError('staged launcher members differ from release')


def predecessor(binding,iteration_ref,actor,*,read,pin):
    """Resolve a previous replacement from its frozen receipts, not the initial client."""
    def need(ok,msg):
        if not ok:raise ValueError(msg)
    previous=read(binding['transition']); area=Path(binding['transition']['path']).parent
    need(previous['iteration_ref']==iteration_ref and pin(area/'actor-client/OWNER.json')==binding['owner'],
         'different previous iteration or owner')
    reader=read(previous['actor_reader_release_ref']);owner=read(binding['owner']);lock=read(reader['source_lock'])
    if 'recovery' in binding:
        need(set(binding)=={'transition','recovery','owner'} and previous['idle_recovery_ref']==binding['recovery']
             and pin(area/'RECOVERY.json')==binding['recovery'],'different idle recovery')
        rec=read(binding['recovery']);old=read(rec['previous_owner']);proof=rec['evidence']
        need(old['resident']==owner['resident'] and old['target']==rec['predecessor']['owner']
             and proof['last_sequence']==rec['predecessor']['sequence'] and proof['checkpoint']==rec['checkpoint']
             and proof['closure']==rec['predecessor']['closure_sha256'] and proof['exit']['exit_code']!=0
             and old['target']!=owner['target'],'unrelated retired actor')
        sw=dict(command=rec['predecessor'],checkpoint=rec['checkpoint'],restore_template=rec['template'])
    else:
        need(set(binding)=={'transition','switch','owner'} and pin(area/'ACTOR-SWITCH.json')==binding['switch'],
             'different previous actor switch')
        sw=read(binding['switch']);ended=read(sw['ended'])
        need(ended['checkpoint_ref']==sw['checkpoint'] and ended['previous']['sha256']==sw['command']['closure_sha256'],
             'unrelated predecessor end')
    stage=read(sw['restore_template'])
    need(reader['iteration_ref']==iteration_ref and reader['source_lock']==owner['resident']['source_lock_ref']
         and lock['precision_lock_sha256']==actor['continuous_actor']['precision_lock_sha256']
         and owner['resident']==dict(actor['continuous_actor'],source_lock_ref=reader['source_lock'])
         and sw['command']['source_boot']==actor['source_boot'],'unrelated predecessor switch')
    if 'launcher_ref' in reader:
        verify_launcher(reader,read=read,pin=pin)
        sw=dict(sw,launcher_ref=reader['launcher_ref'])
    client=area/'actor-client'
    # The active owner's very first submitted batch must actually have resumed this checkpoint.
    first=Path(owner['directory']);call=read(pin(first/'RESIDENT-CALL.json'));req=read(read(pin(first/'SUBMIT-INTENT.json'))['training_ref'])
    need(call==dict(sequence=0,target=owner['target']) and req['restore_command']==sw['command']
         and req['restore_checkpoint_sha256']==sw['checkpoint']['sha256'] and req['continuous_actor']==owner['resident']
         and stage['capture_mounts']['root']==req['capture_mounts']['root'],'first command does not match the previous switch')
    return client,sw,reader['source_lock']


def restore_template(template,area):
    import copy,hashlib
    stage=copy.deepcopy(template);m=stage['capture_mounts'];key='closed_actor_command'
    roots=m['logical_roots'];roles=m['formal_input_roles']
    if len(roles)!=len(set(roles)) or (key in roles)!=(key in roots):raise ValueError('ambiguous restore roles')
    if key in roots and roots[key]!=str(area):
        # Ancestor references remain resolvable on repeated worker replacements.
        alias='ancestor_command_'+hashlib.sha256(roots[key].encode()).hexdigest()[:20]
        if alias in roots or alias in roles:raise ValueError('conflicting ancestor mount')
        roots[alias]=roots[key];roles.append(alias)
    if key not in roots:roles.append(key)
    roots[key]=str(area)
    return stage


def install(config,spec,base):
    import formal_actor_transport as transport
    from dependencies import require
    from run_package import read,pin,write
    new=read(spec['actor_reader_release_ref'])
    old=config['formal_actor']
    oldclient=Path(config['output_root'])/'formal-resident-client'
    if 'predecessor' in new:
        oldclient,prior,lockref=predecessor(new['predecessor'],spec['iteration_ref'],old,read=read,pin=pin)
        require(oldclient.resolve()!=(base/'actor-client').resolve(),'replacement must have a new client ledger')
        config['_resident_client_root']=str(oldclient)
        config['_restored_command']=prior['command'];config['_replacement_checkpoint']=prior['checkpoint']
        old['stage_template_ref']=prior['restore_template']
        if 'launcher_ref' in prior:old['launcher_ref']=prior['launcher_ref']
        old['continuous_actor']=dict(old['continuous_actor'],source_lock_ref=lockref)
    require(new['parent_template']==old['stage_template_ref'] and new['iteration_ref']==spec['iteration_ref'],
            'actor revision targets different experiment')
    template=read(new['stage_template']); lock=read(new['source_lock'])
    require(lock['precision_lock_sha256']==old['continuous_actor']['precision_lock_sha256'],'precision changed')
    if 'launcher_ref' in new:
        verify_launcher(new,read=read,pin=pin)
        require(old['launcher_ref']==new['parent_launcher'] and pin(new['launcher_ref']['path'])==new['launcher_ref']
                and Path(new['launcher_ref']['path']).name=='stage_bounded_actor_step.py','launcher source drift')
    original=transport.train;end_original=transport.end_resident
    switch=base/'ACTOR-SWITCH.json'
    def apply(cfg):
        transition=read(switch);actor=cfg['formal_actor']
        cfg['_restored_command']=transition['command'];cfg['_replacement_checkpoint']=transition['checkpoint']
        cfg['_resident_client_root']=str(base/'actor-client')
        actor['stage_template_ref']=transition['restore_template']
        if 'launcher_ref' in new:actor['launcher_ref']=new['launcher_ref']
        actor['continuous_actor']=dict(actor['continuous_actor'],source_lock_ref=new['source_lock'])
    def train(cfg,state,directory,guard):
        if not switch.exists() and read(state['capture']['input_ref'])['capture'].get('training_input_layout')=='group_refs_v1':
            if (directory/'RESIDENT-CALL.json').exists():
                # A controller handoff is legal while the predecessor is still executing.
                # Reconcile the original signed command on its original client, once.
                call=read(directory/'RESIDENT-CALL.json');intent=read(directory/'SUBMIT-INTENT.json')
                ack=read(directory/'COMMAND-ACK.json');owner=read(oldclient/'OWNER.json')
                import hashlib,json
                wire=hashlib.sha256(json.dumps(intent['command'],sort_keys=True,allow_nan=False).encode()).hexdigest()
                require(ack['remote']==dict(sequence=call['sequence'],source_boot=old['source_boot'],
                            run_id=old['continuous_actor']['run_id'],sha256=wire) and call==dict(target=owner['target'],sequence=intent['sequence'])
                        and intent['target']==owner['target'] and read(ack['command_ref'])==intent['command']
                        and ack['remote']['sequence']==call['sequence'] and intent['source_boot']==old['source_boot']
                        and intent['command']['generation_inventory_ref']==state['capture']['generation_inventory_ref']
                        and intent['command']['checkpoint_sha256']==state['resume_ref']['sha256'],
                        'existing actor command belongs to another source')
                return original(cfg,state,directory,guard)
            require(not (directory/'SUBMIT-INTENT.json').exists() and not (directory/'RESIDENT-CALL.json').exists(),
                    'unreconciled train intent at replacement')
            records=sorted((oldclient/'completed').glob('*.json'));command=read(records[-1]);cp=state['resume_ref']
            require(command['owner']==read(oldclient/'OWNER.json')['target'] and
                    command['sequence']==len(records)-1 and command['checkpoint_sha256']==cp['sha256'] and
                    command['source_boot']==old['source_boot'],'different terminal actor command')
            area=Path(command['owner'])/'hot/resident/results'/format(command['sequence'],'06d')
            require(Path(cp['path']).parents[1]==area and read(cp)['step']==state['step'],'not the exact update boundary')
            restore=write(base/'RESTORE-ACTOR.json',restore_template(template,area))
            # The transport returns the end document on recovery and its pin
            # on first completion. Use the same immutable receipt in both paths.
            end_original(cfg,guard)
            ended=pin(oldclient/'END-COMPLETE.json')
            require(ended and read(ended)['checkpoint_ref']['sha256']==cp['sha256'],'old actor closed another checkpoint')
            write(switch,dict(command=command,checkpoint=cp,restore_template=restore,ended=ended))
        if switch.exists():apply(cfg)
        return original(cfg,state,directory,guard)
    transport.train=train
    if switch.exists():apply(config)
