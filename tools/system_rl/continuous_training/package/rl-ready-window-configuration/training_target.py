"""Tie the requested rollout target to the registered learning Clock and DATA sampling."""
NODES=('P','S1','S2','D')
SYSTEM='system'


def check_target(target,learning,clock,sampling):
    if target not in (*NODES,SYSTEM):raise ValueError('unknown training target')
    step=clock['successful_optimizer_steps']
    from paired_experiment import digest
    if type(step) is not int or step<0 or clock['config_sha256']!=digest(learning):
        raise ValueError('Clock belongs to different learning configuration')
    phases=[p for p in learning['phases'] if p['start_step']<=step]
    if not phases:raise ValueError('no active training phase')
    phase=phases[-1];system=target==SYSTEM
    scope='system' if system else 'node'
    if (phase['rollout_scope']!=scope or phase['target_nodes']!=([] if system else [target])
        or phase['trainable_mask']!={n:(system or n==target) for n in NODES}):
        raise ValueError('requested target differs from active learning phase')
    if sampling['split']!='TRAIN' or sampling.get('scope')!=scope or sampling.get('node')!=(None if system else target):
        raise ValueError('requested target differs from TRAIN roster')
    return dict(target=target,rollout_scope=scope,phase=phase['id'],step=step,trainable_mask=phase['trainable_mask'])
