"""Release a failed session after a committed update and before any successor generation."""
import argparse,fcntl,json
from pathlib import Path
from dependencies import require
from run_package import read,pin,write
from formal_rollout import settings,lane_providers,provider_modules,release


def close(iteration,proof_ref):
    cfg=read(iteration);cfg['_path']=str(iteration);root=Path(cfg['output_root'])
    with (root/'iteration.guard').open('rb') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'CURSOR.json');step=state['step'];s,p=settings(cfg);lanes=lane_providers(cfg,s,p)
        last=state['consumed_cohorts'][-1];capture=state['capture'] if state.get('capture') else last
        require(state['phase']=='endpoint' and step>0 and not state.get('pending_stage') and
                last['result']['checkpoint_ref']==state['resume_ref'] and last['result']['optimizer_steps']==1,
                'not a committed update awaiting activation')
        evidence=read(proof_ref)
        area=root/f'policy-{step:06d}';m=read(area/'MARKER.json')
        require(evidence['engine_switch'] is False and not (area/'ACTIVATIONS-CLOSED.json').exists() and
                not list(area.glob('*/LANE.json')),'switch already acknowledged; reconcile exact owners')
        session=root/'resident-lease';held=read(session/'REMOTE-LEASE.json')
        captures=sorted((root/'rollout').glob('*-capture'))
        terminals=[read(c/'DRAINED.json') for c in captures]
        require(terminals and all(t['unknown_requests']==0 and t['terminal_verified'] for t in terminals),
                'unclosed generation')
        observed={}
        with provider_modules() as (lease,_,_,_):
            remote=held['remote']
            def fetch(args):return lease.command(remote,args,capture_output=True,text=True).stdout
            marker=json.loads(fetch(['cat',m['marker']]))
            require(marker['successor']==evidence['successor'],'tested a different marker')
            for name,provider in lanes.items():
                lane=read(provider['initial_lane_ref']);cid=read(provider['service_ready'])['container_id']
                require(fetch(['docker','inspect','-f','{{.Id}} {{.State.Running}}',cid]).strip()==cid+' true','different service')
                current=json.loads(fetch(['docker','exec',cid,'cat',provider['live_receipt_paths']['activation_ref']]))
                require(current==read(lane['activation_ref']) and current['policy_ref']==marker['previous'],'service state changed')
                metrics=fetch(['curl','-fsS','http://127.0.0.1:'+str(provider['remote_port'])+'/metrics']).splitlines()
                for k in ('running','waiting'):
                    values=[float(l.rsplit(' ',1)[1]) for l in metrics if l.startswith('vllm:num_requests_'+k+'{')]
                    require(values and not any(values),'in-flight generation')
                observed[name]=dict(container_id=cid,initial_activation_ref=lane['activation_ref'])
        receipt=write(root/'ACTIVATION-REJECTED.json',dict(iteration_ref=pin(iteration),cursor_ref=pin(root/'CURSOR.json'),
                checkpoint_ref=state['resume_ref'],policy_ref=state['policy_ref'],charge=state['charged_tokens_upper'],
                sampled_consumed=state['sampled_consumed'],terminal_refs=[pin(c/'DRAINED.json') for c in captures],
                checked_successor_ref=pin(proof_ref),observed=observed,unknown_requests=0))
        term=dict(terminal_verified=True,unknown_requests=0,rejected_activation_ref=receipt)
        return write(session/'RELEASED.json',release(session,held['remote'],read(p['service_config']),term,s['drain_timeout_seconds']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('iteration',type=Path);p.add_argument('reader',type=Path)
    a=p.parse_args();print(close(a.iteration,a.reader))
