"""Run disjoint warm-up, unprofiled measurement and traced windows after both cutovers."""
import json,subprocess,sys,time,urllib.request
from pathlib import Path
from measure_existing_services import run
from io_utils import save,require


def main(config):
    root=Path(config['output']);deadline=time.monotonic()+config['ready_timeout_seconds']
    if config.get('after_rounds'):
        previous=Path(config['after_rounds'])
        while not (previous/'DRAINED.json').is_file():
            require(time.monotonic()<deadline,'previous comparison did not finish');time.sleep(3)
        require(json.loads((previous/'RESULT.json').read_bytes())['passed'] and
                json.loads((previous/'DRAINED.json').read_bytes())['guards_releasable'],'previous comparison failed')
    lazy=config.get('activate_on_first_warmup',{})
    if lazy:
        require(lazy['phase']==config['phases'][0], 'lazy activation only during discarded warm-up')
    for phase in config['phases']:
        if 'phase_specs' in config:
            from io_utils import doc
            seed=doc(config['phase_specs'][phase])
            require(seed['measurement_id']==phase,'different phase spec')
        else:
            seed=json.loads((root/(phase+'.json')).read_bytes())
        for service in seed['services']:
            while True:
                require(time.monotonic()<deadline,'paired service readiness timeout')
                try:
                    cid=Path(config['start_receipts'][service['name']]) if 'start_receipts' in config else root/'cutover'/('STARTED-'+service['endpoint'].rsplit(':',1)[1]+'.json')
                    if not cid.exists():time.sleep(4);continue
                    target=json.loads(cid.read_bytes())['container_id']
                    state=json.loads(subprocess.check_output(['docker','inspect',service['container']]))[0]
                    require(state['Id']==target and state['State']['Running'],'candidate exited or identity changed')
                    with urllib.request.urlopen(service['endpoint']+'/v1/models',timeout=5) as res:
                        models=json.load(res)['data']
                    if any(x['id']==config['adapter_alias'] and x.get('parent')==config['base_model'] for x in models):break
                    if (phase==lazy.get('phase') and service['name']==lazy['lane']
                            and 'stock_adapter_alias' not in service and [x['id'] for x in models]==[config['base_model']]):break
                except OSError:pass
                time.sleep(5)
            pid=state['State']['Pid']
            service.update(pid=pid,start_ticks=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]))
        if 'start_receipts' in config:
            seed['start_receipts']=config['start_receipts']
        path=root/phase/'CONFIG.local.json';save(path,seed);run(path)
        if phase==lazy.get('phase'):
            s,=[s for s in seed['services'] if s['name']==lazy['lane']]
            raw=subprocess.check_output(['docker','exec',s['container'],'cat','/runtime/state/policy/ACTIVATION.json'])
            receipt=json.loads(raw)
            require(receipt['alias']==receipt['policy_version']==config['adapter_alias']
                    and 'checkpoint-sha256:'+receipt['weights_ref']['sha256']==config['adapter_alias']
                    and receipt['service_identity']['container_name']==s['container'], 'warm-up active policy differs')
if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_bytes()))
