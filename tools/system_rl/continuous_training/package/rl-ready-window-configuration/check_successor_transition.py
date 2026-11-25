"""Rehearse a second in-place reader replacement against the original signed ledgers (no SSH or allocation)."""
import argparse,copy,json,sys,tempfile
from pathlib import Path

def run(release,work,scratch):
    import formal_actor_transport as transport
    from run_package import read,pin,write
    from install_indexed_actor import install,restore_template
    from dependencies import require
    spec=read(release/'TRANSITION.json');client=release/'actor-client';cfg=read(spec['iteration_ref'])
    parent=read(release/'ACTOR-SWITCH.json');owner=read(client/'OWNER.json')
    req=read(read(work/'SUBMIT-INTENT.json')['training_ref']);call=read(work/'RESIDENT-CALL.json')
    prev=read(client/'completed'/f"{call['sequence']-1:06d}.json")
    cp=read(work.parent.with_name('step-'+format(req['prior_step'],'06d'))/'update/returned/RETURN.json')['checkpoint_ref']
    cap=read(req['generation']['pending_capture_ref'])
    state=dict(step=req['prior_step'],resume_ref=cp,capture=dict(input_ref=read(cap['pending_capture_ref'])['input_ref'],
                      generation_inventory_ref=req['generation']['generation_inventory_ref']))
    bind={k:pin(release/f) for k,f in [('transition','TRANSITION.json'),('switch','ACTOR-SWITCH.json'),('owner','actor-client/OWNER.json')]}
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        area=Path(tmp)
        reader=read(spec['actor_reader_release_ref'])
        rev=dict(reader,parent_template=parent['restore_template'],stage_template=parent['restore_template'],predecessor=bind)
        spec=dict(spec,actor_reader_release_ref=write(area/'CANDIDATE.json',rev))
        original=transport.train;original_end=transport.end_resident
        seen=[]
        def reconcile(config,st,d,g):
            require(config['_resident_client_root']==str(client) and config['formal_actor']['stage_template_ref']==parent['restore_template']
                and config['_restored_command']==parent['command'] and d==work and st['resume_ref']==cp,'adopted wrong worker')
            seen.append(call);return 'reconcile'
        def never(*a):raise AssertionError('unexpected actor end during replay')
        transport.train=reconcile;transport.end_resident=never
        try:
            install(cfg,spec,area)
            require(transport.train(cfg,state,work,None)=='reconcile','command lost')
            bad=copy.deepcopy(state);bad['resume_ref']['sha256']='0'*64
            try:transport.train(cfg,bad,work,None)
            except ValueError:pass
            else:raise ValueError('stale command accepted')
            require(seen==[call] and not (area/'ACTOR-SWITCH.json').exists(),'unexpected switch')
            where=Path(cp['path']).parents[1];m=restore_template(read(parent['restore_template']),where)
            require(m==restore_template(m,where) and prev['owner']==owner['target'] and prev['checkpoint_sha256']==cp['sha256'],'incorrect next anchor')
            roots=m['capture_mounts']['logical_roots'];roles=m['capture_mounts']['formal_input_roles']
            require(roots['closed_actor_command']==str(where) and len(roles)==len(set(roles)) and
                    set(read(parent['restore_template'])['capture_mounts']['logical_roots'].values())<=set(roots.values()),'lost ancestor')
            return dict(signed_sequence=call['sequence'],preceding_step=state['step'],old_owner_adopted=True,stale_command_rejected=True,
                        idempotent_mounts=True,previous_checkpoint_addressable=True,dispatches=0)
        finally:transport.train=original;transport.end_resident=original_end

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('release','update','scratch'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.release,a.update,a.scratch)))
