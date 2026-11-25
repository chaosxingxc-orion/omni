"""Resume a closed checkpoint after idle exit, only if the next update never reached the old worker."""
import argparse, hashlib, json, os, shlex, subprocess
from pathlib import Path
from prepare_transition_config import pin
from prepare_recording_macros import read,seal

PROBE="""
import hashlib,json,subprocess,sys
from pathlib import Path
q=json.load(sys.stdin);p=Path(q['owner']);hot=p/'hot';r=hot/'resident';seq=q['sequence']
def get(p):return json.loads(p.read_bytes())
def need(ok):
 if not ok:raise ValueError('unproven idle exit or pending command')
need(p.is_absolute() and p==p.resolve() and seq>0 and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==q['boot'])
ex=get(hot/'EXIT.json');cid=ex['container_id'];obs=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
need(obs['Id']==cid==get(hot/'CONTAINER.json')['container_id'] and obs['State']['Status']=='exited' and ex['exit_code']==obs['State']['ExitCode']!=0)
need('resident idle timeout; preserve last full checkpoint' in subprocess.check_output(['docker','logs','--tail','400',cid],stderr=subprocess.STDOUT).decode(errors='replace'))
need(not (r/'END.json').exists() and max(int(f.stem) for f in (r/'commands').glob('*.json'))==seq-1 and max(int(f.name) for f in (r/'results').iterdir() if f.is_dir())==seq-1)
last=r/'results'/format(seq-1,'06d');cp=Path(q['checkpoint']['path'])
need(cp.parents[1]==last and get(last/'RESULT.json')['checkpoint_ref']==q['checkpoint'] and hashlib.sha256(cp.read_bytes()).hexdigest()==q['checkpoint']['sha256'] and hashlib.sha256((last/'CLOSED.json').read_bytes()).hexdigest()==q['closure'])
print(json.dumps({'exit':ex,'last_sequence':seq-1,'unaccepted_sequence':seq,'checkpoint':q['checkpoint'],'closure':q['closure']}))
"""

def plan(parent,client,source,out):
    from install_indexed_actor import restore_template
    spec=read(pin(parent));cfg=read(spec['iteration_ref']);root=Path(cfg['output_root'])
    cursor=read(pin(root/'CURSOR.json'));step=cursor['cohort_sequence'];u=root/f'step-{step:06d}'/'update'
    from report_training_updates import controller_guard_owner
    if controller_guard_owner(root,Path(spec['iteration_ref']['path'])) is not None:raise ValueError('controller is still active')
    owner=read(pin(client/'OWNER.json'));call=read(pin(u/'RESIDENT-CALL.json'));lease=read(pin(u/'SUBMIT-INTENT.json'))
    i=call['sequence'];last=read(pin(client/'completed'/f'{i-1:06d}.json'));cmd=read(pin(u/'COMMAND.json'))
    cp=cursor['resume_ref'];rr=read(spec['actor_reader_release_ref']);old=read(pin(client.parent/'ACTOR-SWITCH.json'))
    if not (cursor['phase']=='update' and cursor['pending_stage']['stage']=='update' and i>0 and
            cfg['formal_actor']['source_boot']==lease['source_boot']==last['source_boot'] and
            cmd==lease['command'] and cmd['action']=='update' and cmd['sequence']==lease['sequence']==i and
            last['owner']==call['target']==lease['target']==owner['target'] and read(pin(client/'calls'/f'{i:06d}.json'))==dict(directory=str(u),call=call) and
            cmd['run_id']==owner['resident']['run_id']==cfg['run_id'] and rr['source_lock']==owner['resident']['source_lock_ref'] and
            cmd['previous_close_sha256']==last['closure_sha256'] and cmd['checkpoint_sha256']==last['checkpoint_sha256']==cp['sha256'] and
            cmd['generation_inventory_ref']==cursor['capture']['generation_inventory_ref'] and
            cmd['pending_capture_ref']==cursor['capture']['pending_capture_ref'] and not any((u/n).exists() for n in ('COMMAND-ACK.json','returned','closed-return.tar'))):
        raise ValueError('not an unacknowledged command for this checkpoint')
    inv=read(cmd['generation_inventory_ref']);drain=read(pin(Path(inv['cohort_root']).parent/'DRAINED.json'))
    if not(inv['closed'] and drain['terminal_verified'] and drain['unknown_requests']==0 and drain['generation_closed_ref']==inv['generation_closed_ref']):raise ValueError('unclosed capture')
    cfg0=read(lease['training_ref']);q=dict(owner=owner['target'],sequence=i,boot=lease['source_boot'],checkpoint=cp,closure=last['closure_sha256'])
    evidence=json.loads(subprocess.check_output(['ssh','-oBatchMode=yes',cfg0['host'],shlex.join(['python3','-B','-c',PROBE])],input=json.dumps(q).encode(),timeout=40))
    out.mkdir(parents=True,exist_ok=True)
    area=Path(cp['path']).parents[1]
    template=restore_template(read(old['restore_template']),area)
    binding=seal(out/'RECOVERY.json',dict(previous_owner=pin(client/'OWNER.json'),predecessor=last,checkpoint=cp,
        cursor=pin(root/'CURSOR.json'),original=pin(u/'SUBMIT-INTENT.json'),generation=cmd['generation_inventory_ref'],
        seq=step,directory=str(u),evidence=evidence,template=seal(out/'STAGE.json',template)))
    for name in ('transition_controller','recover_idle_actor'):
        dest=out/(name+'.py');data=(source/dest.name).read_bytes()
        if dest.exists() and dest.read_bytes()!=data:raise ValueError('different staged source')
        if not dest.exists():dest.write_bytes(data)
        spec['sources'][name]=pin(dest)
    spec['idle_recovery_ref']=binding
    return seal(out/'TRANSITION.json',spec)


def install(config,spec):
    from run_package import read as verified,write
    import formal_actor_transport as actor
    import formal_actor_inputs as inputs
    r=verified(spec['idle_recovery_ref']);base=Path(spec['idle_recovery_ref']['path']).parent
    owner=verified(r['previous_owner']);original=Path(r['directory'])
    if verified(r['original'])['command']['generation_inventory_ref']!=r['generation']:raise ValueError('different pending generation')
    config['_resident_client_root']=str(base/'actor-client')
    config['_restored_command']=r['predecessor'];config['_replacement_checkpoint']=r['checkpoint']
    config['formal_actor'].update(stage_template_ref=r['template'],continuous_actor=owner['resident'])
    train=actor.train;stage=inputs.stage
    recovered=original/'idle-successor'/'update'
    def stage_inputs(template,cfg,directory):
        if directory==recovered:
            import copy
            template=copy.deepcopy(template);m=template['capture_mounts'];m['formal_input_root']+='-idle-successor'
        return stage(template,cfg,directory)
    def resume(cfg,state,directory,guard):
        if directory!=original:return train(cfg,state,directory,guard)
        if (state['resume_ref']!=r['checkpoint'] or state['capture']['generation_inventory_ref']!=r['generation']
            or (original/'COMMAND-ACK.json').exists()):raise ValueError('unrelated update at idle recovery')
        result=train(cfg,state,recovered,guard)
        loc=recovered/'returned'
        # The ordinary restart scan uses the canonical returned graph.
        dest=original/'returned'
        if not dest.is_symlink():
            if dest.exists():raise ValueError('different result at original attempt')
            os.symlink(loc.relative_to(original),dest,target_is_directory=True)
            fd=os.open(original,os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        if dest.resolve(strict=True)!=loc.resolve(strict=True):raise ValueError('different returned graph')
        write(original/'IDLE-RECOVERY.json',dict(binding=spec['idle_recovery_ref'],result=result))
        return result
    actor.train=resume;inputs.stage=stage_inputs

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for n in ('parent','client','source','out'):cli.add_argument('--'+n,required=True,type=Path)
    print(json.dumps(plan(**vars(cli.parse_args()))))
