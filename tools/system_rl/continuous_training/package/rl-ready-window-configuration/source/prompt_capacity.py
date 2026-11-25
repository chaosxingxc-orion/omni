"""Shared node input capacities for preflight, capture and accounting."""
NODES=('P','S1','S2','D')


def bind_diagnostic_budget(iteration,training_package,spec):
    """Bind one teacher-forcing envelope to admission and actual execution."""
    required=positive(spec['diagnostic_token_upper'],'diagnostic teacher-forcing tokens')
    stage=iteration['stage_budgets']['diagnostic']
    package=training_package['budget']
    envelope=max(required,positive(stage['tokens_upper'],'diagnostic stage tokens'),
                 positive(package['diagnostic_token_upper'],'diagnostic package tokens'))
    stage['tokens_upper']=envelope
    package['diagnostic_token_upper']=envelope
    return envelope


def positive(value,name):
    if type(value) is not int or value<=0:raise ValueError('positive integer '+name)
    return value


def prompt_bound(budget,node):
    fallback=positive(budget['prompt_tokens_upper_per_call'],'prompt bound')
    bounds=budget.get('prompt_tokens_upper_by_node')
    if bounds is None:return fallback
    if set(bounds)!=set(NODES):raise ValueError('all four node prompt bounds required')
    for n,value in bounds.items():positive(value,n+' prompt bound')
    if max(bounds.values())!=fallback:raise ValueError('maximum node prompt bound differs')
    if node not in bounds:raise ValueError('unknown node')
    return bounds[node]


def completion_bound(budget,node):
    fallback=positive(budget['completion_tokens_upper_per_call'],'completion bound')
    bounds=budget.get('completion_tokens_upper_by_node')
    if bounds is None:return fallback
    if set(bounds)!=set(NODES):raise ValueError('all four node completion bounds required')
    for n,value in bounds.items():positive(value,n+' completion bound')
    if max(bounds.values())>fallback:raise ValueError('node completion exceeds scalar envelope')
    if node not in bounds:raise ValueError('unknown node')
    return bounds[node]


def request_cap(budget,packet):
    output=positive(packet['wire']['max_tokens'],'completion cap')
    if budget.get('completion_tokens_upper_by_node') is not None:
        if output>completion_bound(budget,packet.get('stage')):
            raise ValueError('actual request exceeds configured node completion bound')
    return prompt_bound(budget,packet.get('stage'))+output


def diagnostic_work(spec):
    b=spec['bounds'];groups=positive(b['groups'],'groups');replicas=positive(b['G'],'G')
    counts=dict(P=1,S1=b['S1_folds'],S2=b['S2_types_upper'],D=1)
    selection=spec.get('packet_selection')
    if selection:
        if spec.get('input_mode')!='frozen_node_inputs':
            raise ValueError('bounded diagnostics require frozen node inputs')
        counts=diagnostic_quotas(selection)
    else:
        for node,count in counts.items():positive(count,node+' packet count')
    completions={n:completion_bound(b,n) for n in NODES}
    calls=groups*replicas*sum(counts.values())
    tokens=groups*replicas*sum(counts[n]*(prompt_bound(b,n)+completions[n]) for n in NODES)
    context=spec.get('context_limit')
    if context is not None:
        positive(context,'consumer context')
        if max(prompt_bound(b,n)+completions[n] for n in NODES)>context:
            raise ValueError('diagnostic input/output exceeds consumer context')
    return calls,tokens


def diagnostic_quotas(selection):
    """One explicit sampling scope shared by accounting and actual dispatch."""
    if set(selection)!={'schema','inputs_per_node_per_draw'}:
        raise ValueError('explicit diagnostic packet quota fields')
    schema=selection['schema'];counts=selection['inputs_per_node_per_draw']
    if schema not in ('diagnostic-packet-selection-v1','diagnostic-packet-selection-v2'):
        raise ValueError('diagnostic selection schema')
    lower=1 if schema.endswith('v1') else 0
    if set(counts)!=set(NODES) or not all(type(q) is int and lower<=q<=55 for q in counts.values()):
        raise ValueError('invalid diagnostic node quota')
    if not any(counts.values()):raise ValueError('empty diagnostic scope')
    return dict(counts)


def sampled_envelope(profile, budget, recordings, replicas, scope, node=None):
    """A recording/group is the sampling unit. Reserve for physical model calls."""
    episodes = positive(recordings, 'window recordings') * positive(replicas, 'G')
    if profile['schema'] == 'sampled-node-window-capacity-v1':
        if scope != 'node' or node != 'P' or profile['node'] != node:
            raise ValueError('legacy single-call capacity is specific to P')
        limits = {node: (1, positive(profile['completion_tokens_upper_per_call'], 'P cap'))}
    else:
        if (profile['schema'] != 'sampled-rollout-capacity-v2' or profile['scope'] != scope
                or scope not in ('node', 'system') or (scope == 'node' and node not in NODES)
                or (scope == 'system' and node is not None)):
            raise ValueError('sampled rollout scope mismatch')
        nodes = profile['nodes']
        if set(nodes) != ({node} if scope == 'node' else set(NODES)):
            raise ValueError('capacity must cover exactly the active nodes')
        limits = {n: (positive(v['calls_per_episode_upper'], n+' calls'),
                      positive(v['completion_tokens_upper'], n+' cap')) for n,v in nodes.items()}
        for n,(calls,cap) in limits.items():
            if (n in ('P','D') and calls != 1) or cap > completion_bound(budget,n):
                raise ValueError('invalid node call or completion ceiling: '+n)
    return (episodes * sum(v[0] for v in limits.values()),
            episodes * sum(calls*(prompt_bound(budget,n)+cap) for n,(calls,cap) in limits.items()),
            {n:cap for n,(_,cap) in limits.items()})
