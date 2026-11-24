"""Close a failed initial branch window only if dispatch never began."""
import argparse,fcntl,sqlite3
from contextlib import ExitStack
from pathlib import Path
from run_package import pin,read,write
from dependencies import require


def close(iteration):
    from formal_rollout import settings,lane_providers,provider_modules,release
    c=read(iteration);c['_path']=str(iteration);root=Path(c['output_root'])
    s,p=settings(c);providers=lane_providers(c,s,p)
    with ExitStack() as stack:
        def lock(path):
            f=stack.enter_context(path.open('rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock(root/'iteration.guard')
        state=read(root/'CURSOR.json');seq=state['pending_stage']['sequence']
        require(c.get('branch_initial_ref') and state['step']==0 and not state['resume_ref'] and
                state['phase']==state['pending_stage']['stage']=='capture','not an initial failed capture')
        r=root/'rollout'/f'{seq:04d}-capture';cohort=r/'cohort';fence=r/'lease'
        lock(cohort/'COHORT.guard')
        for i in range(len(providers)):lock(cohort/f'lane-{i}.guard')
        failure=read(fence/'DRAIN')
        require(failure['reason']=='resident lane failed' and 'validate_publication' in read(failure['failure_ref'])['traceback']
                and not (cohort/'groups').exists() and not (cohort/'GENERATION-CLOSED.json').exists()
                and not any(cohort.rglob('*.dispatch.json')) and not any(cohort.rglob('*.intent.json')),
                'failure is not before packet construction')
        with sqlite3.connect(Path(c['registry']['path']).as_uri()+'?mode=ro',uri=True) as db:
            counts=dict(db.execute('select status,count(*) from spend group by status'))
        require(set(counts)<={'known','closed_unobserved'},'registry contains unresolved requests')
        session=root/'resident-lease';held=read(session/'REMOTE-LEASE.json')
        require(held==read(fence/'REMOTE-LEASE.json') and read(session/'activation/CONFIG.json')['iteration_ref']==pin(iteration),
                'wrong lease for this capture')
        with provider_modules() as (lease,_,_,_):
            for provider in providers.values():
                cid=read(provider['service_ready'])['container_id']
                remote=held['remote']
                obs=lease.command(remote,['docker','inspect','-f','{{.Id}} {{.State.Running}}',cid],capture_output=True,text=True)
                require(obs.stdout.strip()==cid+' true','resident changed')
                metrics=lease.command(remote,['curl','-fsS','http://127.0.0.1:'+str(provider['remote_port'])+'/metrics'],capture_output=True,text=True).stdout.splitlines()
                for k in ('running','waiting'):
                    values=[float(l.rsplit(' ',1)[1]) for l in metrics if l.startswith('vllm:num_requests_'+k+'{')]
                    require(values and not any(values),'resident still busy')
        receipt=write(root/'UNDISPATCHED.json',dict(iteration_ref=pin(iteration),cursor_ref=pin(root/'CURSOR.json'),
                failure_ref=failure['failure_ref'],selection_ref=pin(cohort/'SELECTED.json'),window_ref=state['sampled_pending_window_ref'],
                charged_tokens_upper=state['charged_tokens_upper'],registry=counts,unknown_requests=0,business_requests=0,optimizer_steps=0))
        terminal=dict(terminal_verified=True,unknown_requests=0,undispatched_ref=receipt)
        return write(session/'RELEASED.json',release(session,held['remote'],read(p['service_config']),terminal,s['drain_timeout_seconds']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('iteration',type=Path);print(close(p.parse_args().iteration))
