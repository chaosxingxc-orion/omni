"""Extend the existing component scorer in its original consumer import scope.

No reward/kernel/trainer fork: P/S2/D and all aggregation stay upstream.
S1 scores a complete type set with F2; unknown type labels are explicitly masked.
"""
from contextlib import contextmanager
import sys
from dependencies import require


def extend(scoring):
    if getattr(scoring, '_dp8_f2', False):
        return
    scoring._dp8_f2 = True
    scoring.BOUNDED_METRICS['S1'].update(('f2','exact'))
    original = scoring._score_valid
    def valid(name, online, output, labels):
        if name != 'S1' or labels is None:
            return original(name, online, output, labels)
        universe = set(online['type_universe'])
        require(len(universe) == len(online['type_universe']) and universe, 'explicit type universe required')
        annotated = set(labels.get('annotated_types', []))
        gold = set(labels['present_types'])
        require(annotated <= universe and gold <= annotated, 'unknown labels cannot become negatives')
        if not annotated:
            return dict(mask=0, metrics={}, reason='no_annotated_types')
        pred = output['present_types']
        require(isinstance(pred,list) and len(set(pred)) == len(pred) and set(pred) <= universe, 'invalid predicted type set')
        part = original(name, dict(type_universe=sorted(annotated)),
                        dict(present_types=sorted(set(pred)&annotated)),
                        dict(annotated_types=sorted(annotated),present_types=sorted(gold)))
        m = part['metrics']; tp,fp,fn = (m[k] for k in ('tp','fp','fn'))
        m.update(f2=scoring.ratio(5*tp,5*tp+4*fn+fp), exact=int(fp == fn == 0),
                 count_error=abs(len(set(pred)&annotated)-len(gold)),
                 annotated_types=len(annotated), masked_types=len(universe-annotated))
        return part
    scoring._score_valid = valid
    # The original fold union iterates type_universe. Restrict each fold to its
    # explicit annotation mask before that union; unknown predictions stay audit-only.
    original_episode = scoring.score_episode
    def episode(value, definition):
        import copy
        item = value.get('offline',{}).get('S1')
        if item is not None:
            # Only S1's offline annotation mask is rewritten. Audio and the
            # sampled actions are read-only inputs to the original scorer.
            item = copy.deepcopy(item)
            value = dict(value, offline=dict(value['offline'], S1=item))
            items = item.get('calls',[item])
            for row in items:
                labels = row.get('labels')
                if labels is None: continue
                annotated = set(labels.get('annotated_types',[]))
                universe = set(row['online']['type_universe'])
                require(annotated <= universe and set(labels['present_types']) <= annotated, 'annotation mask mismatch')
                if not annotated:
                    row['labels'] = None
                    continue
                if len(items)>1:
                    row['online']['type_universe'] = sorted(annotated)
                    output = row['output']
                    if isinstance(output,dict) and isinstance(output.get('present_types'),list):
                        pred = output['present_types']
                        if len(set(pred)) == len(pred) and set(pred)<=universe:
                            output['present_types'] = [t for t in pred if t in annotated]
                        else:
                            output['terminal'] = 'model_invalid'
        return original_episode(value,definition)
    scoring.score_episode = episode
    # learning.py imported score_episode by value before this scoped extension.
    sys.modules['learning'].score_episode = episode


def install(task):
    consumer = task.consumer
    if getattr(consumer, '_dp8_f2', False): return
    consumer._dp8_f2 = True
    original = consumer.unified
    @contextmanager
    def scope(config):
        with original(config) as result:
            extend(sys.modules['scoring'])
            configure_clip(result[1])
            configure_frozen(result[1])
            yield result
    consumer.unified = scope
    consumer.legacy.unified = scope


def configure_clip(schedule):
    """Scoped admission seam: scalar clipped_token already has clip parameter."""
    if getattr(schedule,'_dp8_clip',False):return
    import inspect,types
    old=schedule.check_config
    text=inspect.getsource(old)
    anchor="k['clip'] == .2"
    require(text.count(anchor)==1,'clip admission source changed')
    text=text.replace(anchor,"(type(k['clip']) in (int,float) and math.isfinite(k['clip']) and 0<k['clip']<1)")
    scope=dict(old.__globals__)
    exec(compile(text,inspect.getfile(old)+':configured-clip','exec'),scope)
    schedule.check_config=scope['check_config']
    # Clock.__init__ resolves the original module globals, not its attributes
    # on a SimpleNamespace consumer wrapper.
    old.__globals__['check_config']=scope['check_config']
    schedule._dp8_clip=True


def configure_frozen(schedule):
    if getattr(schedule,'_dp8_frozen',False):return
    import inspect
    old=schedule.check_config;text=inspect.getsource(old) if not getattr(schedule,'_dp8_clip',False) else None
    # configure_clip function is exec-generated; its source is retained by
    # rebuilding from the pinned original check_config, with the same seams.
    from dependencies import ROOT
    import ast
    tree=ast.parse((ROOT/'unified-learning/schedule.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_config')
    text=ast.unparse(fn)
    text=text.replace("k['clip'] == 0.2", "(type(k['clip']) in (int,float) and math.isfinite(k['clip']) and 0<k['clip']<1)")
    anchor="routing['frozen_upstream_adapter_vector'] == k['behavior_policy_vector']"
    require(text.count(anchor)==1,'frozen vector admission seam changed')
    text=text.replace(anchor,"(set(routing['frozen_upstream_adapter_vector'])==set(k['behavior_policy_vector']) and all(isinstance(v,str) and v for v in routing['frozen_upstream_adapter_vector'].values()))")
    namespace=dict(old.__globals__);exec(compile(text,str(ROOT/'unified-learning/schedule.py')+':phase-frozen','exec'),namespace)
    schedule.check_config=namespace['check_config'];old.__globals__['check_config']=namespace['check_config'];schedule._dp8_frozen=True
