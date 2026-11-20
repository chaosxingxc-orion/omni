"""One public, data-only local-profile resolver for RL/collection/cleanup.

${NAME} substitutes an explicit profile value; entire-value substitutions retain
JSON types. No environment lookup, Python evaluation, or shell interpolation.
"""
import argparse,copy,hashlib,json,re
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

def compile_profile(template_path,profile_path,output):
    from run_package import read,pin,write
    from storage import environment
    template=read(template_path);profile=read(profile_path);resolved=resolve(template,profile)
    root=Path(output);environment(root,0)
    # Keep explicit input snapshots under local durable storage, not an archive.
    template_ref=write(root/'TEMPLATE.json',template);profile_ref=write(root/'PROFILE.local.json',profile)
    run_ref=write(root/'RESOLVED.json',resolved)
    return write(root/'RESOLUTION.json',dict(schema='omni-profile-resolution-v1',
        template_source_ref=pin(template_path),profile_source_ref=pin(profile_path),
        template_snapshot_ref=template_ref,profile_snapshot_ref=profile_ref,resolved_ref=run_ref,
        template_sha256=digest(template),profile_sha256=digest(profile),resolved_sha256=digest(resolved),
        profile_local_only=True,dispatch=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--profile',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(compile_profile(a.template,a.profile,a.output)))
if __name__=='__main__':main()
