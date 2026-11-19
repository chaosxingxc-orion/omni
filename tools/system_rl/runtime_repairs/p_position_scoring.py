"""Offline-only versioned component scores; definitions are engineering candidates."""
from collections import Counter
import math
from kernel import require, number, digest

VERSION = 'component-definitions-v3-p-position'
LEGACY_VERSION = 'component-definitions-v2'
COMPONENTS = ('P', 'S1', 'S2', 'D', 'system')
DEFINITIONS = {
    'P': 'gold intent rank in unique legal retrieved top3: 1/0.75/0.5; absent 0; coverage reported separately',
    'S1': 'binary type existence over explicit annotated universe; union overlapping annotated types and predictions across folds before precision/recall/FN/FP rates',
    'S2': 'sum per-call multiset exact (type,value) TP/FP/FN on value-annotated supplied types before precision/recall/F1; null legal abstention',
    'D': 'joint exact intent and multiset slot decision on actual frozen upstream',
    'system': 'per-example terminal joint exact SLU surrogate; not corpus official micro-F1',
}

BOUNDED_METRICS={
    'P':{'coverage_at3','coverage_at1','coverage_at2','position_reward_v1'},'S1':{'precision','recall','fn_rate','fp_rate'},
    'S2':{'precision','recall','f1'},'D':{'joint_exact','intent_correct','precision','recall','f1'},
    'system':{'joint_exact','intent_correct','precision','recall','f1'}}


def reward_bounds(weights,name):
    require(set(weights) <= BOUNDED_METRICS[name] and weights, 'reward mixture requires registered metrics bounded in [0,1]')
    return sum(min(0,number(v,'metric coefficient')) for v in weights.values()),sum(max(0,number(v,'metric coefficient')) for v in weights.values())


def validate_definition(definition):
    require(definition['version'] in (VERSION,LEGACY_VERSION), 'unknown scorer definition version')
    if definition['version']==LEGACY_VERSION:
        require(set(definition['metric_weights'].get('P',{})) <= {'coverage_at3'},
                'position reward requires explicit v3 definition')
    policy=definition['failure_policy']
    require(policy['version']=='component-lower-bound-v1' and policy['partial_branch_failure']=='whole_component_floor', 'explicit supported failure/partial policy required')
    require(set(policy['component_values']) == set(COMPONENTS), 'all failure values explicit')
    for name,weights in definition['metric_weights'].items():
        lower,upper=reward_bounds(weights,name)
        require(number(policy['component_values'][name],'failure value') <= lower, 'observed failure reward exceeds derived valid lower bound')


def ratio(n, d, empty=1.0):
    return n / d if d else empty


def value_counts(slots, allowed=None):
    require(isinstance(slots, dict), 'slot map required')
    result = Counter()
    for typ, values in slots.items():
        require(isinstance(typ, str) and isinstance(values, list), 'type/list values required; null is a list member')
        require(all(v is None or isinstance(v, str) for v in values), 'invalid value')
        if allowed is None or typ in allowed:
            result.update((typ, v) for v in values if v is not None)
    return result


def multiset_metrics(pred, gold):
    tp, fp, fn = sum((pred & gold).values()), sum((pred - gold).values()), sum((gold - pred).values())
    return dict(tp=tp, fp=fp, fn=fn, precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                f1=ratio(2*tp, 2*tp+fp+fn))


def _score_valid(name, online, output, labels):
    """Missing supervision masks a component, never renormalizes remaining weights."""
    require(name in COMPONENTS, 'unknown reward component')
    if labels is None:
        return dict(mask=0, metrics={}, reason='missing_supervision')
    if name == 'P':
        candidates, selected = online['retrieved_candidates'], output['selected']
        require(isinstance(candidates, list) and len(set(candidates)) == len(candidates), 'invalid topK')
        require(isinstance(selected, list) and len(selected) <= 3 and len(set(selected)) == len(selected)
                and set(selected) <= set(candidates), 'P must select unique topK -> at most top3')
        require(isinstance(labels['intent'], str), 'intent label required')
        rank=selected.index(labels['intent'])+1 if labels['intent'] in selected else 0
        metrics = dict(coverage_at3=int(rank>0),coverage_at1=int(rank==1),
                       coverage_at2=int(0<rank<=2),
                       position_reward_v1={0:0.0,1:1.0,2:0.75,3:0.5}[rank])
    elif name == 'S1':
        universe, gold, pred = online['type_universe'], labels['present_types'], output['present_types']
        require(len(set(universe)) == len(universe) and universe, 'S1 explicit annotated universe required')
        require(set(labels['annotated_types']) == set(universe), 'partial S1 existence annotations must be masked explicitly')
        require(len(set(pred)) == len(pred) and set(pred) <= set(universe) and set(gold) <= set(universe), 'S1 types outside universe')
        gold, pred, universe = set(gold), set(pred), set(universe)
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        metrics = dict(tp=tp, fp=fp, fn=fn, tn=len(universe - pred - gold),
                       precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                       fn_rate=ratio(fn, len(gold), 0), fp_rate=ratio(fp, len(universe-gold), 0))
    elif name == 'S2':
        supplied = set(online['supplied_types'])
        mask = labels['value_annotation_mask']
        require(set(mask) == supplied and all(type(v) is bool for v in mask.values()), 'explicit value annotation mask for every supplied type')
        require(set(output['slots']) == supplied, 'S2 must return supplied types (null allowed)')
        annotated = {k for k in supplied if mask[k]}
        if not annotated:
            return dict(mask=0, metrics={}, reason='no_annotated_supplied_values')
        require(annotated <= set(labels['slots']), 'annotated value absent from labels')
        metrics = multiset_metrics(value_counts(output['slots'], annotated), value_counts(labels['slots'], annotated))
        metrics.update(annotated_types=len(annotated), masked_types=len(supplied-annotated))
    else:
        require(labels.get('joint_labels_observable') is True, 'joint decision labels unavailable')
        require(output.get('terminal') in ('valid_final', 'model_invalid', 'model_length', 'protocol_error'), 'unknown is not failure')
        valid = output['terminal'] == 'valid_final'
        if valid:
            metrics = multiset_metrics(value_counts(output['slots']), value_counts(labels['slots']))
            metrics['intent_correct'] = int(output['intent'] == labels['intent'])
            metrics['joint_exact'] = int(metrics['intent_correct'] and metrics['fp'] == 0 and metrics['fn'] == 0)
        else:
            metrics = dict(joint_exact=0, intent_correct=0)
    return dict(mask=1, metrics=metrics)


def score_component(name,online,output,labels):
    if labels is None:
        return dict(mask=0,metrics={},reason='missing_supervision')
    # Validate inputs/supervision separately; malformed model output is an observed
    # failure, whereas missing or inconsistent evidence must reject admission.
    if name == 'P':
        dummy=dict(selected=[])
    elif name == 'S1':
        dummy=dict(present_types=[])
    elif name == 'S2':
        dummy=dict(slots={t:[None] for t in online['supplied_types']})
    else:
        dummy=dict(intent=labels['intent'],slots=labels['slots'],terminal='valid_final')
    baseline=_score_valid(name,online,dummy,labels)
    if not baseline['mask']:
        return baseline
    status=output.get('terminal','valid_final') if isinstance(output,dict) else 'model_invalid'
    require(status in ('valid_final','model_invalid','model_length','protocol_error'), 'unavailable transport/evidence is not observed model failure')
    try:
        require(status == 'valid_final', 'observed model terminal failure')
        part=_score_valid(name,online,output,labels)
        return part
    except (ValueError,KeyError,TypeError):
        metrics={key:0 for key in baseline['metrics']}
        for key in ('fn','fn_rate','annotated_types','masked_types'):
            if key in baseline['metrics']:metrics[key]=baseline['metrics'][key]
        return dict(mask=1,metrics=metrics,observed_model_failure=True,reason=status if status!='valid_final' else 'malformed_or_empty_model_output')


def score_episode(episode, definition):
    validate_definition(definition)
    result = {}
    for name, metric_weights in definition['metric_weights'].items():
        require(name in COMPONENTS and metric_weights, 'empty/unknown component metrics')
        item = episode.get('offline', {}).get(name)
        if item is None:
            result[name] = dict(mask=0, value=0.0, metrics={}, reason='not_sampled_or_not_supervised')
            continue
        items = item.get('calls', [item])
        require(items and all(isinstance(i,dict) for i in items), 'empty component call list')
        if name == 'S2' and len(items)>1:
            owners=set()
            for i in items:
                labels=i.get('labels')
                if labels is None:continue
                occurrence_ids=labels.get('value_occurrence_ids')
                require(isinstance(occurrence_ids,dict), 'multi-call S2 requires actual value occurrence ownership')
                for typ,annotated in labels['value_annotation_mask'].items():
                    if not annotated:continue
                    values=labels['slots'][typ]
                    ids=occurrence_ids.get(typ)
                    require(isinstance(ids,list) and len(ids) == sum(v is not None for v in values), 'S2 value/occurrence roster mismatch')
                    require(all(isinstance(oid,str) and oid for oid in ids) and len(set(ids))==len(ids), 'invalid S2 occurrence identity')
                    require(not owners.intersection(ids), 'same gold occurrence assigned to multiple S2 calls')
                    owners.update(ids)
        parts = [score_component(name, i['online'], i['output'], i.get('labels')) for i in items]
        observed = [p for p in parts if p['mask']]
        if name == 'S1' and len(items)>1 and observed:
            universe=set();gold=set();pred=set();annotations={}
            for i,p in zip(items,parts):
                if not p['mask']:continue
                u=set(i['online']['type_universe']);g=set(i['labels']['present_types'])
                for typ in u:
                    require(typ not in annotations or annotations[typ] == (typ in g), 'conflicting labels across S1 folds')
                    annotations[typ]=typ in g
                universe.update(u);gold.update(g)
                if not p.get('observed_model_failure'):pred.update(i['output']['present_types'])
            metrics=_score_valid('S1',dict(type_universe=sorted(universe)),dict(present_types=sorted(pred)),
                                 dict(annotated_types=sorted(universe),present_types=sorted(gold)))['metrics']
            scored=dict(mask=1,metrics=metrics)
        elif not observed:
            scored = dict(mask=0, metrics={}, reason='missing_supervision')
        elif len(parts) == 1:
            scored = parts[0]
        else:
            require(name in ('S1','S2'), 'P/D/system require one actual decision call')
            metrics = {k:sum(p['metrics'][k] for p in observed) for k in ('tp','fp','fn')}
            tp,fp,fn = (metrics[k] for k in ('tp','fp','fn'))
            metrics.update(precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn))
            if name == 'S2':
                metrics.update(f1=ratio(2*tp,2*tp+fp+fn),
                               annotated_types=sum(p['metrics']['annotated_types'] for p in observed),
                               masked_types=sum(p['metrics'].get('masked_types',0) for p in parts),
                               masked_calls=len(parts)-len(observed))
            else:
                tn=sum(p['metrics']['tn'] for p in observed)
                metrics.update(tn=tn,fn_rate=ratio(fn,tp+fn,0),fp_rate=ratio(fp,fp+tn,0))
            scored=dict(mask=1,metrics=metrics)
        scored['call_provenance']=[dict(call_ids=i['call_ids'],mask=p['mask'],metrics=p['metrics'],observed_model_failure=p.get('observed_model_failure',False)) for i,p in zip(items,parts)]
        scored['observed_model_failure']=any(p.get('observed_model_failure',False) for p in parts)
        if scored['mask']:
            require(set(metric_weights) <= set(scored['metrics']), 'metric not defined for component')
            scored['value'] = math.fsum(number(w, 'metric coefficient') * scored['metrics'][m] for m, w in metric_weights.items())
        else:
            scored['value'] = 0.0
        if scored['observed_model_failure']:
            scored['value']=definition['failure_policy']['component_values'][name]
            scored['failure_policy']=definition['failure_policy']['partial_branch_failure']
        scored['derived_valid_reward_bounds']=reward_bounds(metric_weights,name)
        result[name] = scored
    return result


def reward(components, snapshot):
    return math.fsum(snapshot['weights'][name] * part['mask'] * part['value'] for name, part in components.items())
