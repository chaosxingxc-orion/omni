"""CPU sidecar after queue owner's capture; does not launch or acquire a lease."""
import argparse
import time
from train_runtime import preflight
from common import *

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--state',required=True)
    p.add_argument('--output',required=True);p.add_argument('--original-launch-epoch',type=float)
    a=p.parse_args();cfg,packets=preflight(a.config,ROOT)
    require(cfg['runtime']['status']=='OWNER_RECONCILED','UNBOUND runtime: main review and queue capture first')
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
    class Journal(ContactJournal):
        def __init__(self):
            self.root=Path(a.state)/'contact-journals'/cfg['run_id']/cfg['attempt_id']
            self.identity=dict(run_id=cfg['run_id'],attempt_id=cfg['attempt_id'],arm='provisional-S-only-fit',
                protocol_hash=sha(canonical({k:v for k,v in cfg.items() if k!='runtime'})),split_identity_hash=sha(canonical([r['sample_id'] for r,_ in packets])))
    journal=Journal()
    states=[journal.inspect('S-'+sha(canonical([cfg['run_id'],cfg['attempt_id'],r['sample_id'],r['wire_sha256']])))['state'] for r,_ in packets]
    require(all(s in ('absent','observed') for s in states),'UNKNOWN_NO_RESEND')
    resumed=states.count('observed')>0
    if resumed:
        terminal=read(Path(a.state)/'TERMINAL.json')
        require(terminal['status'] in ('PAUSED','FAILED','COMPLETE') and terminal['unknown']==0,'old runner must exit and drain before resume')
        require(a.original_launch_epoch is not None,'retain original launch epoch and cost')
    save(a.output,dict(schema='s-execution-efficiency-v1',run_id=cfg['run_id'],attempt_id=cfg['attempt_id'],
        original_runtime_config_sha256=sha(Path(a.config).read_bytes()),executor_max_workers=cfg["concurrency"],
        runner_sha256=sha((Path(__file__).parent/'train-runtime-v1/run_s_native.py').read_bytes()),
        original_runner_sha256=sha((Path(__file__).parent/'train-runtime-v1/run_s_native.py').read_bytes()),
        resources_module_sha256=sha((Path(__file__).parent/'train-runtime-v1/native_resources.py').read_bytes()),
        branch_adapter_sha256=sha((PACKAGE/'code/run.py').read_bytes()),
        resumed=resumed,original_launch_epoch=a.original_launch_epoch or time.time()))
    print('EXECUTION_SIDECAR_BOUND_NOT_LAUNCHED')
