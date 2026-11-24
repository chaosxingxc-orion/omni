"""Original consumer with selected draws, complete no-gradient and official masks."""
import contextlib
import copy
import math
import types
from cohort_io import BASE,load,read,raw,require,digest

legacy=load(BASE/'training-consumer-successor/adapter.py','fresh_original_consumer')
CONSUMER_KIND=legacy.CONSUMER_KIND
CONSUMER_ID=legacy.CONSUMER_ID

def bind_function(fn,**overrides):
    return types.FunctionType(fn.__code__,dict(fn.__globals__,**overrides),fn.__name__,fn.__defaults__,fn.__closure__)

@contextlib.contextmanager
def unified(config):
    with legacy.unified(config) as (learning,schedule,kernel):
        # Zero trainable calls are a legal zero contribution after full evidence
        # validation. The original scalar reducer already supports empty sums.
        namespace=dict(learning.__dict__)
        source=(BASE/'unified-learning/learning.py').read_text()
        copied="adapted = copy.deepcopy(g)"
        require(source.count(copied)==1,'group projection changed')
        # evaluate only replaces each episode's calls list when masking nodes;
        # reuse its immutable tensor lists and create new group/episode dicts.
        source=source.replace(copied,"adapted = dict(g, episodes=[dict(e) for e in g['episodes']])")
        anchor="            require(ae['calls'], 'no sampled trainable calls')"
        require(source.count(anchor)==1,'original frozen-call admission seam changed')
        anchor="            require(all(components[n]['mask'] == 1 for n in targets), 'target supervision missing; no node admission')"
        require(source.count(anchor)==1,'original supervision seam changed')
        source=source.replace("            require(ae['calls'], 'no sampled trainable calls')", "            # Frozen calls retain original group/episode denominator.")
        source=source.replace(anchor,"            # Missing official values retain mask=0 and the full denominator.")
        anchor="                require(e.get('joint_labels_observable') is True, 'failed system episode still needs offline label observability')"
        require(source.count(anchor)==1,'original early-failure observability seam changed')
        source=source.replace(anchor,"                # Unavailable official slot values are masked, never fabricated.")
        source=source.replace("components['system']=dict(mask=1,value=c['scorer']['failure_policy']['component_values']['system']", "components['system']=dict(mask=int(e.get('joint_labels_observable') is True),value=c['scorer']['failure_policy']['component_values']['system']")
        exec(compile(source,str(BASE/'unified-learning/learning.py')+':fresh-official-mask','exec'),namespace)
        scoped=types.SimpleNamespace(**learning.__dict__)
        scoped.evaluate=namespace['evaluate']
        yield scoped,schedule,kernel

def validate_draw_binding(files,config,rows=None):
    plan=read(config['selected_cohort_ref'])
    require(plan['schema']=='selected-fresh-cohort-v1','selected cohort schema')
    if rows is None:rows={n:read(files['refs'][n]) for n in ('online','offline','provenance')}
    require(files['count']==len(plan['draws'])==len(rows['provenance']),'all selected draws required; no resampling')
    gids=set();qids=set()
    for i,(d,p,o) in enumerate(zip(plan['draws'],rows['provenance'],rows['online'],strict=True)):
        expected={k:v for k,v in d.items() if k!='record'}
        expected.update({k:d['record'][k] for k in ('sample_id','recording_ref_id','source_id','source','domain','outcome')})
        require(p==expected,'draw probabilities/identity/outcome changed')
        g=o['behavior_bundle']['groups'][0]
        require(g['group_id']==d['group_id'] and g['query_draw_id']==d['query_draw_id'] and g['query_id']==d['record']['sample_id'],'fresh group belongs to another draw')
        require(g['group_id'] not in gids and g['query_draw_id'] not in qids,'duplicate fresh group reuse')
        gids.add(g['group_id']);qids.add(g['query_draw_id'])
    require(gids==set(plan['group_ids']),'selected group roster differs')
    return plan

def restore(files,config,session=None):
    if not config.get('selected_cohort_ref'):return legacy.restore(files,config,session)
    validate_draw_binding(files,config)
    source=(BASE/'training-consumer-successor/adapter.py').read_text()
    anchor="        require(prov['duplicate_occurrence'] == 1, 'fresh group reuse forbidden')"
    require(source.count(anchor)==1,'original consumer duplicate seam changed')
    namespace=dict(legacy.__dict__)
    exec(compile(source.replace(anchor,"        # Exact draw + independent fresh group checked by successor."),str(BASE/'training-consumer-successor/adapter.py')+':fresh','exec'),namespace)
    return namespace['restore'](files,config,session)

def build(files,config,session=None):
    return bind_function(legacy.build,restore=restore,unified=unified)(files,config,session)

def retained_shape(value):
    """Validate a JSON tensor in rows, without allocating a list per scalar."""
    if not isinstance(value,list):
        require(type(value) in (float,int) and math.isfinite(value),'invalid tensor scalar')
        return []
    if not value:
        return [0]
    if not isinstance(value[0],list):
        require(all(type(v) in (float,int) and math.isfinite(v) for v in value),
                'invalid tensor scalar or ragged retained tensor')
        return [len(value)]
    dims=retained_shape(value[0])
    require(all(retained_shape(child)==dims for child in value[1:]),'ragged retained tensor')
    return [len(value),*dims]

def build_with_bundle(files,config):
    """One consumer acceptance: validate selected draws on its freshly read rows.

    Build and publication share the same restored bundle only for this call.
    The historical adapter's checks, report and tensor construction stay intact.
    """
    require(config.get('selected_cohort_ref'),'one-pass build requires selected cohort')
    captured=[]
    def restore_once(files,config,session):
        require(not captured,'consumer requested more than one restore')
        # Same fresh-group seam as restore(); run its draw validation on
        # the actual pin-checked inputs immediately before the original join.
        src=(BASE/'training-consumer-successor/adapter.py').read_text()
        duplicate="        require(prov['duplicate_occurrence'] == 1, 'fresh group reuse forbidden')"
        before_join='    indexes = {}'
        require(src.count(duplicate)==1 and src.count(before_join)==1,'consumer restore seam changed')
        src=src.replace(before_join,'    validate_draw_binding(files,config,rows)\n'+before_join)
        src=src.replace(duplicate,'        # Fresh draw and group identity validated above.')
        # These objects just came from json.loads of authenticated bytes in
        # this restore. Transfer sole ownership when joining offline labels.
        clone="b = copy.deepcopy(online['behavior_bundle'])"
        bins="shape(f['audio_features'])[1]"
        require(src.count(clone)==1 and src.count(bins)==2,
                'consumer bundle/feature validation seam changed')
        src=src.replace(clone,"b = online.pop('behavior_bundle')")
        # The validator compared every actual shape with its declaration above.
        src=src.replace(bins,"specs['audio_features']['shape'][1]")
        ns=dict(legacy.__dict__,validate_draw_binding=validate_draw_binding)
        exec(compile(src,str(BASE/'training-consumer-successor/adapter.py')+':selected-once','exec'),ns)
        # restore() privately deserializes the rows before joining supervision.
        # They have no other consumer once draw binding has passed.
        ns['shape']=retained_shape
        result=ns['restore'](files,config,session)
        captured.append(result[0])
        return result
    fields,report=bind_function(legacy.build,restore=restore_once,unified=unified)(files,config)
    require(len(captured)==1,'consumer did not restore selected data')
    return fields,report,captured[0]


def validate_dataset(*,files,config):return build(files,config)[1]
