"""Move an accepted consumer batch between hosts without building another dataset."""
from pathlib import Path
import sys
import torch


def build_packet(dataset_root,*,accepted=None):
    from cohort_io import read,pin,require,digest
    receipt=read(pin(dataset_root/'COMPLETE.json'))
    require(receipt['training_ready'] and not receipt['synthetic_fixture'],'require accepted real TRAIN')
    if accepted is None:
        import consumer
        from fresh_adapter import consumer as configure
        fields,report=configure(consumer).build(read(receipt['files_ref']),read(receipt['consumer_config_ref']))
    else:
        fields,report=accepted
    require(digest(report)==digest(read(pin(dataset_root/'CONSUMER-REPORT.json'))),'packed batch differs from accepted cohort')
    native=read(receipt['native_config_ref']); learning=read(native['learning_config_ref'])
    attempt=native.get('attempt_id')
    require(isinstance(attempt,str) and attempt.strip(), 'missing native optimizer attempt identity')
    return dict(tensors=dict(fields['tensors']),non_tensors={k:list(v) for k,v in fields['non_tensors'].items()},meta=fields['meta_info'],
        report=report,sidecar=read(native['task_sidecar_ref']),online=read(native['online_by_call_ref']),
        clip=learning['kernel']['clip'],clock=read(native['clock_state_ref']),learning=learning,consumer=read(native['consumer_config_ref']),
        selected_groups=read(native['selected_cohort_ref'])['group_ids'],
        attempt_id=attempt,dataset=pin(dataset_root/'COMPLETE.json'))


def seal_packet(root,fields,report,inventory,*,packet_directory=None):
    """Publish payload and manifest together; interrupted staging never replaces a commit."""
    import os,tempfile
    from native_artifacts import save_torch
    from run_package import write,pin
    from dependencies import require
    base=packet_directory or root
    target=base/'actor-packet'
    if target.exists():
        prior=load_packet(root,inventory,packet_directory=base)
        require(prior['report']==report,'existing packet belongs to another acceptance')
        return pin(target/'RECEIPT.json')
    packet=build_packet(root,accepted=(fields,report))
    tmp=Path(tempfile.mkdtemp(prefix='.actor-packet-',dir=base))
    payload=save_torch(tmp/'fields.pt',packet)
    write(tmp/'RECEIPT.json',dict(schema='accepted-actor-packet-v1',dataset=packet['dataset'],
        report=pin(root/'CONSUMER-REPORT.json'),inventory=inventory,
        payload={k:payload[k] for k in ('bytes','sha256')}))
    os.rename(tmp,target)
    fd=os.open(base,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return pin(target/'RECEIPT.json')


def load_packet(root,inventory,*,packet_directory=None):
    """Authenticate bytes before deserialization; scope is exactly one sampled cohort."""
    from run_package import read,pin
    from dependencies import require
    target=(packet_directory or root)/'actor-packet'
    record=read(target/'RECEIPT.json');path=target/'fields.pt'
    require(record['schema']=='accepted-actor-packet-v1' and record['inventory']==inventory and
            record['dataset']==pin(root/'COMPLETE.json') and record['report']==pin(root/'CONSUMER-REPORT.json'),
            'sealed packet does not belong to this generation')
    ref=pin(path)
    require(record['payload']=={k:ref[k] for k in ('bytes','sha256')},'sealed packet bytes changed')
    packet=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(packet['dataset']==record['dataset'] and packet['report']['required_fields_validated'] and
            not packet['meta']['synthetic'],'invalid accepted packet')
    return packet


def pack(c,parents):
    sys.path.insert(0,str(parents/'jev-system-rl/implementation'))
    from fresh_adapter import scope
    with scope():
        from cohort_io import pin
        packet=build_packet(Path(c['prepared'])/c['dataset_version'])
        target=Path(c['prepared'])/c['pack_fields']
        if target.parent!=Path(c['prepared']):raise ValueError('invalid pack path')
        with target.open('xb') as f:torch.save(packet,f)
        print('packed',packet['report']['actor_rows'],packet['report']['action_tokens'],pin(target)['sha256'])
