"""Move an accepted consumer batch between hosts without building another dataset."""
from pathlib import Path
import sys
import torch


def pack(c,parents):
    sys.path.insert(0,str(parents/'jev-system-rl/implementation'))
    from fresh_adapter import scope
    with scope() as cohort:
        from cohort_io import read,pin,require,digest
        ds=Path(c['prepared'])/c['dataset_version'];receipt=read(pin(ds/'COMPLETE.json'))
        require(receipt['training_ready'] and not receipt['synthetic_fixture'],'require accepted real TRAIN')
        import consumer
        from fresh_adapter import consumer as configure
        fields,report=configure(consumer).build(read(receipt['files_ref']),read(receipt['consumer_config_ref']))
        require(digest(report)==digest(read(pin(ds/'CONSUMER-REPORT.json'))),'packed batch differs from accepted cohort')
        native=read(receipt['native_config_ref']); learning=read(native['learning_config_ref'])
        ns=fields['non_tensors']
        packet=dict(tensors=dict(fields['tensors']),non_tensors={k:list(v) for k,v in ns.items()},meta=fields['meta_info'],
            report=report,sidecar=read(native['task_sidecar_ref']),online=read(native['online_by_call_ref']),
            clip=learning['kernel']['clip'],clock=read(native['clock_state_ref']),learning=learning,consumer=read(native['consumer_config_ref']),
            selected_groups=read(native['selected_cohort_ref'])['group_ids'],dataset=pin(ds/'COMPLETE.json'))
        target=Path(c['prepared'])/c['pack_fields']
        if target.parent!=Path(c['prepared']):raise ValueError('invalid pack path')
        with target.open('xb') as f:torch.save(packet,f)
        print('packed',report['actor_rows'],report['action_tokens'],pin(target)['sha256'])
