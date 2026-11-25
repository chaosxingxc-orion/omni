"""Select an independent node task for the existing single-adapter worker.

Each branch has its own run/Clock/optimizer. This compiles a task, not a scheduler
or multi-adapter inference backend. Frozen base initialization is explicit.
"""
import copy
from dependencies import require

NODES=('P','S1','S2','D')


def shift_curve(spec, origin, curve):
    """Preserve the exact existing curve evaluated at local_step + origin."""
    kind=spec['kind']
    if kind=='hold':return copy.deepcopy(spec)
    if kind=='piecewise':
        points=[[0,curve(spec,origin)]]
        points += [[step-origin,value] for step,value in spec['points'] if step>origin]
        return dict(spec,points=points)
    require(kind in ('linear','cosine'),'known coefficient schedule required')
    if spec['end_step']<=origin:return dict(kind='hold',value=curve(spec,origin))
    # Cosine curves crossing phase entry cannot be truncated into an equivalent
    # new cosine. Preserve them via an explicit offset wrapper would change the
    # existing scheduler; reject that case instead of silently changing math.
    require(spec['start_step']>=origin or kind=='linear','cross-phase cosine requires explicit local schedule')
    return dict(spec,start_step=max(0,spec['start_step']-origin),end_step=spec['end_step']-origin,
                start_value=curve(spec,origin) if spec['start_step']<origin else spec['start_value'])


def select(learning, spec, *, mode):
    if spec is None:return learning
    require(mode=='formal','a configured training task is not diagnostic acceptance')
    system = spec.get('schema')=='system-training-phase-v1'
    if system:
        require(set(spec)=={'schema','branch_id','initialization'} and spec['initialization']=='locked_composition',
                'system optimization requires a locked composition')
        node=None
    else:
        require(set(spec)=={'schema','node','branch_id','initialization'} and
                spec['schema']=='independent-node-training-v1' and spec['node'] in NODES and
                spec['initialization']=='frozen_base','explicit independent node/base identity required')
        node=spec['node']
    require(isinstance(spec['branch_id'],str) and bool(spec['branch_id']),'task identity required')
    result=copy.deepcopy(learning)
    matches=[p for p in result['phases'] if
             (p['rollout_scope']=='system' and not p['target_nodes'] if system else
              p['rollout_scope']=='node' and p['target_nodes']==[node])]
    require(len(matches)==1,'one original training phase required')
    phase=matches[0];origin=phase['start_step'];phase['start_step']=0
    require(phase['trainable_mask']=={n:system or n==node for n in NODES},'incorrect training mask')
    if system:
        require(phase['required_validations'],'system promotion gate required')
    else:
        phase['required_validations']=[]
    from dependencies import ROOT,upstream
    from run_package import pin
    pins={n:pin(ROOT/'unified-learning'/n) for n in ('kernel.py','scoring.py','schedule.py','learning.py','dataset.py')}
    pins.update({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    with upstream().consumer.unified(dict(source_pins=pins)) as (_,schedule,_):
        for key in ('loss_node_weights','reward_weights'):
            phase[key]={k:shift_curve(v,origin,schedule.curve) for k,v in phase[key].items()}
        losses=result['task_loss']['phases'][phase['id']]
        result['task_loss']['phases']={phase['id']:{k:shift_curve(v,origin,schedule.curve) for k,v in losses.items()}}
    result['phases']=[phase]
    require(result['adapter_routing']['mode']=='shared' and
            set(result['adapter_routing']['trainable_adapters'].values())=={'default'},
            'branch owns one actual default adapter; composition is separate')
    result['version']+='@independent-'+spec['branch_id']
    return result
