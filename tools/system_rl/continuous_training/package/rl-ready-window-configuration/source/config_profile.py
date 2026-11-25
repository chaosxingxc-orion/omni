"""One public, data-only local-profile resolver for RL/collection/cleanup.

${NAME} substitutes an explicit profile value; entire-value substitutions retain
JSON types. Deployment profiles bind declared environment values once at startup.
"""
import argparse,copy,fcntl,hashlib,json,os,re
from pathlib import Path
MACRO=re.compile(r'\$\{([A-Z][A-Z0-9_]*)\}')
NAME=re.compile(r'[A-Z][A-Z0-9_]*\Z')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def resolve(template,profile):
    if profile.get('schema')!='omni-local-profile-v1':raise ValueError('local profile schema')
    values=profile['values']
    if not isinstance(values,dict) or not all(NAME.fullmatch(k) for k in values):raise ValueError('macro names must be explicit uppercase identifiers')
    def visit(value,stack=()):
        if isinstance(value,dict):return {k:visit(v,stack) for k,v in value.items()}
        if isinstance(value,list):return [visit(v,stack) for v in value]
        if not isinstance(value,str):return value
        def lookup(name):
            if name not in values:raise ValueError('undefined local profile macro: '+name)
            if name in stack:raise ValueError('cyclic local profile macro: '+' -> '.join((*stack,name)))
            return visit(copy.deepcopy(values[name]),(*stack,name))
        match=MACRO.fullmatch(value)
        if match:return lookup(match[1])
        def replace(m):
            v=lookup(m[1])
            if isinstance(v,(dict,list)) or v is None:raise ValueError('structured macro must occupy entire JSON value: '+m[1])
            return str(v)
        out=MACRO.sub(replace,value)
        if '${' in out:raise ValueError('malformed macro syntax')
        return out
    return visit(template)

def stage_config(template,profile,stage):
    """Resolve only the requested stage: later receipt macros need not exist yet."""
    if template.get('schema')!='recording-screen-experiment-v1':
        return resolve(template,profile)
    if set(template)!={'schema','parameters','stages'} or stage not in template['stages']:
        raise ValueError('missing or invalid recording experiment stage: '+stage)
    params=template['parameters']
    if not isinstance(params,dict) or params.keys() & profile['values'].keys():
        raise ValueError('experiment and deployment macros overlap')
    values={**profile['values'],**params}
    if stage=='s1-fold-panel-v1' and 'RECALL_RESULTS_SHA256' not in values and isinstance(values.get('RECALL'),str):
        values['RECALL_RESULTS_SHA256']=values['RECALL']  # old panel-only site profile
    combined=dict(profile,values=values)
    cfg=resolve(template['stages'][stage],combined)
    if cfg.get('schema')!=stage:raise ValueError('different configured stage: '+stage)
    return cfg


def typed_value(name, value, kind, from_env):
    if from_env and kind in ('int', 'port', 'devices'):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError('invalid ' + kind + ' for ' + name) from None
    if kind == 'path':
        valid = isinstance(value, str) and Path(value).is_absolute() and chr(0) not in value
    elif kind == 'host':
        valid = isinstance(value, str) and re.fullmatch(r'(?:[a-zA-Z0-9_][\w.-]*@)?[a-zA-Z0-9_][\w.-]*', value)
    elif kind == 'port':
        valid = type(value) is int and 1 <= value <= 65535
    elif kind == 'int':
        valid = type(value) is int
    elif kind == 'devices':
        valid = (isinstance(value, list) and len(value) > 0
                 and all(type(n) is int and n >= 0 for n in value) and len(set(value)) == len(value))
    else:
        raise ValueError('unknown deployment type for ' + name)
    if not valid or (isinstance(value, str) and '${' in value):
        raise ValueError('invalid ' + kind + ' for ' + name)
    return value


def bind_environment(profile, environ):
    """Only explicitly declared deployment names may read the process environment."""
    if profile.get('schema') != 'omni-local-profile-v2':
        raise ValueError('expected omni-local-profile-v2')
    bindings = profile['environment']
    values = copy.deepcopy(profile['values'])
    if not isinstance(bindings, dict) or not isinstance(values, dict):
        raise ValueError('expected deployment bindings and values')
    effective = dict(schema='omni-local-profile-v1', values=values)
    sources = {}
    for name, kind in bindings.items():
        if not NAME.fullmatch(name):
            raise ValueError('invalid deployment macro name')
        key = 'OMNI_DEPLOY_' + name
        if key in environ:
            values[name] = typed_value(name, environ[key], kind, True)
            sources[name] = key
        else:
            sources[name] = 'profile'
    # Resolve defaults after applying all overrides so derived paths follow the new root.
    for name, kind in bindings.items():
        candidate = resolve('${' + name + '}', effective)
        values[name] = typed_value(name, candidate, kind, False)
    return effective, sources


def compile_deployment(template_ref, profile_ref, root):
    from run_package import pin, read, write

    # A partial attempt can resume from the first durable binding. Concurrent starts
    # cannot bind two environments to one run directory.
    with (root / 'resolution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = dict(template_source_ref=template_ref, profile_source_ref=profile_ref)
        binding = root / 'ENVIRONMENT.json'
        template = read(template_ref)
        profile = read(profile_ref)
        if binding.exists():
            saved = read(binding)
            if saved['inputs'] != inputs:
                raise ValueError('deployment inputs changed for existing run')
            effective = saved['effective_profile']
        else:
            effective, sources = bind_environment(profile, os.environ)
            resolve(template, effective)
            write(binding, dict(inputs=inputs, effective_profile=effective, sources=sources))
        resolved = resolve(template, effective)
        return write(root / 'RESOLUTION.json', dict(
            schema='omni-profile-resolution-v2', **inputs, environment_ref=pin(binding),
            template_snapshot_ref=write(root / 'TEMPLATE.json', template),
            original_profile_ref=write(root / 'PROFILE.input.json', profile),
            profile_snapshot_ref=write(root / 'PROFILE.local.json', effective),
            resolved_ref=write(root / 'RESOLVED.json', resolved),
            template_sha256=digest(template), profile_sha256=digest(effective),
            resolved_sha256=digest(resolved), profile_local_only=True, dispatch=False))


def compile_profile(template_path,profile_path,output):
    from run_package import read,pin,write
    from storage import environment
    template_ref=pin(template_path); profile_ref=pin(profile_path)
    template=read(template_ref); profile=read(profile_ref)
    root=Path(output)
    if profile.get('schema')=='omni-local-profile-v2':
        environment(root,0)
        return compile_deployment(template_ref,profile_ref,root)
    resolved=resolve(template,profile)
    environment(root,0)
    # Keep explicit input snapshots under local durable storage, not an archive.
    template_ref=write(root/'TEMPLATE.json',template);profile_ref=write(root/'PROFILE.local.json',profile)
    run_ref=write(root/'RESOLVED.json',resolved)
    return write(root/'RESOLUTION.json',dict(schema='omni-profile-resolution-v1',
        template_source_ref=pin(template_path),profile_source_ref=pin(profile_path),
        template_snapshot_ref=template_ref,profile_snapshot_ref=profile_ref,resolved_ref=run_ref,
        template_sha256=digest(template),profile_sha256=digest(profile),resolved_sha256=digest(resolved),
        profile_local_only=True,dispatch=False))

def main():
    p=argparse.ArgumentParser()
    for name,env in [('template','OMNI_DEPLOYMENT_TEMPLATE'),('profile','OMNI_DEPLOYMENT_PROFILE'),('output','OMNI_DEPLOYMENT_OUTPUT')]:
        default=os.environ.get(env)
        p.add_argument('--'+name,default=default,required=not default,help='defaults to '+env)
    a=p.parse_args()
    print(json.dumps(compile_profile(a.template,a.profile,a.output)))
if __name__=='__main__':main()
