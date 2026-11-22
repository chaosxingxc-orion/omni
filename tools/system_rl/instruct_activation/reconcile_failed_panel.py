"""Seal a failed paired panel only if every planned HTTP outcome is already terminal."""
import argparse, fcntl, hashlib, json, subprocess
from contextlib import ExitStack
from pathlib import Path
from io_utils import canonical, doc, pin, require, save
from measure_existing_services import distinct_rounds, idle, metrics, service_identity


def inspect_window(path, starts):
    c=doc(pin(path)); root=Path(c['output']); begin=doc(pin(root/'STARTED.json'))
    require(doc(begin['config_ref'])==c and c['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'window identity changed')
    plans=list(distinct_rounds(c,c['services']))
    require(len(plans)==1 and not list(root.rglob('UNKNOWN.json')) and not (root/'RESULT.json').exists(),'expected one failed fully known round')
    _,_,roster,_=plans[0]; outcomes=[]; exited={}; live={}; paths=set()
    for s in c['services']:
        name=s['name']; tag=s['container']; start=doc(pin(starts[name]))
        state=json.loads(subprocess.check_output(['docker','inspect',tag]))[0]
        require(state['Id']==start['container_id'],'container replaced')
        if state['State']['Running']:
            service_identity(s); idle(metrics(s['endpoint'])); live[tag]=state['Id']
        else:
            require(state['State']['Status']=='exited' and not state['State']['Restarting'],'not exited')
            exited[tag]=state['Id']
        for g,i,w in roster[name]:
            if 'stock_adapter_alias' in s:w=dict(w,model=s['stock_adapter_alias'])
            d=root/'round-0'/name/f'{g}-{i}'; ip=d/'INTENT.json'; resp=d/'RESPONSE.json'
            intent=doc(pin(ip)); r=doc(pin(resp)); paths.add(d)
            require(intent['endpoint']==s['endpoint'] and not intent['automatic_retry'] and
                    intent['wire_sha256']==hashlib.sha256(canonical(w)).hexdigest() and
                    hashlib.sha256(r['raw'].encode()).hexdigest()==r['response_sha256'] and 100<=r['status']<=599,'receipt mismatch')
            outcomes.append(dict(container=tag,intent=pin(ip),response=pin(resp),status=r['status']))
    require(exited and any(o['status']>=400 for o in outcomes) and
            all(paths=={p.parent for p in root.rglob(n)} for n in ('RESPONSE.json','INTENT.json')),'failure scope differs')
    return dict(passed=False,training_use=False,guards_releasable=True,unknown_requests=0,
                exited_containers=exited,idle_containers=live,outcomes=outcomes,started=pin(root/'STARTED.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('run',type=Path)
    p.add_argument('--seal',action='store_true'); a=p.parse_args()
    c=doc(pin(a.config)); run=doc(pin(a.run))
    with ExitStack() as stack:
        if a.seal:
            for guard in sorted(set(c['guards'])):
                f=stack.enter_context(open(guard,'rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=inspect_window(a.config,run['start_receipts'])
        if a.seal:save(Path(c['output'])/'DRAINED.json',result)
        print(json.dumps(dict(outcomes=len(result['outcomes']),exited=list(result['exited_containers']),sealed=a.seal)))
