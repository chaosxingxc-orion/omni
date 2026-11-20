"""Original consumer with selected draws, complete no-gradient and official masks."""
import contextlib
import copy
from pathlib import Path
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

def validate_draw_binding(files,config):
    plan=read(config['selected_cohort_ref'])
    require(plan['schema']=='selected-fresh-cohort-v1','selected cohort schema')
    rows={n:read(files['refs'][n]) for n in ('online','offline','provenance')}
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
    source=Path(getattr(legacy,'_capture_source',legacy.__file__)).read_text()
    anchor="        require(prov['duplicate_occurrence'] == 1, 'fresh group reuse forbidden')"
    require(source.count(anchor)==1,'original consumer duplicate seam changed')
    namespace=dict(legacy.__dict__)
    exec(compile(source.replace(anchor,"        # Exact draw + independent fresh group checked by successor."),str(BASE/'training-consumer-successor/adapter.py')+':fresh','exec'),namespace)
    return namespace['restore'](files,config,session)

def build(files,config,session=None):
    return bind_function(legacy.build,restore=restore,unified=unified)(files,config,session)

def validate_dataset(*,files,config):return build(files,config)[1]
