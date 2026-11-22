"""Prepare an isolated successor from an existing measured study; never starts services."""
import hashlib,json,re,sys
from pathlib import Path

def load(p):return json.loads(p.read_bytes())

def prepare(spec):
    root=Path(spec['root']).resolve(strict=True)
    old,new=spec['predecessor'],spec['study']
    tag=spec['attempt'];lane=spec['lane']
    if not all(re.fullmatch('[a-z0-9][a-z0-9-]{0,64}',s) for s in (old,new,tag,lane)) or old==new:
        raise ValueError('invalid study identities')
    prev=load(root/(old+'-switch.local.json'))
    item,=prev['lanes']
    oldcode=Path(item['new_code']);code=root/new/('lane-'+lane)
    if oldcode!=root/old/('lane-'+lane) or not code.is_dir():raise ValueError('candidate lane missing')
    engine=load(code/'config.json'); container=engine['name']+'-'+tag
    if engine['guard']!=prev['guard']:raise ValueError('guard differs')
    def replace(obj):
        if isinstance(obj,dict):return {k:replace(v) for k,v in obj.items()}
        if isinstance(obj,list):return [replace(v) for v in obj]
        if not isinstance(obj,str):return obj
        if obj.startswith(str(root)+'/') or obj.startswith(old+'-'):return obj.replace(old,new)
        return obj
    def get(n):return replace(load(root/(old+n)))
    result={}
    refs=load(root/(new+'-panel')/'PANEL.json')
    if len(refs)!=len(set(r['sha256'] for r in refs)):raise ValueError('duplicate wire')
    for r in refs:
        p=Path(r['path']).resolve(strict=True)
        if p.parent!=root/(new+'-panel')/'wires':raise ValueError('unexpected wire')
        data=p.read_bytes()
        if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('wire drift')
    used=0
    for phase in ('-warm','-latency'):
        suf=phase+'.json';d=get(suf);panel,=d['distinct_panels'];end=used+len(panel)
        d['distinct_panels']=[refs[used:end]]
        if len(d['distinct_panels'][0])!=len(panel):raise ValueError('incomplete panel')
        used=end
        s,=[s for s in d['services'] if s['name']==lane]
        if s['endpoint'].rsplit(':',1)[-1]!=str(engine['port']):raise ValueError('endpoint changed')
        s['container']=container
        result[suf]=d
    if used!=len(refs):raise ValueError('unused panel')
    cut=get('-switch.local.json')
    cut['lanes']=[dict(old_code=str(oldcode),old_attempt=item['new_attempt'],new_code=str(code),new_attempt=tag,preflight=spec['preflight'])]
    cut['after_score']=str(root/(old+'-score')/'RESULT.json');cut['terminal']=str(Path(cut['after_score']).with_name('DRAINED.json'))
    result['-switch.local.json']=cut
    for suf in ('-run.local.json','-score-input.local.json','-flow.local.json'):result[suf]=get(suf)
    files={root/(new+n):v for n,v in result.items()}
    if any(p.exists() for p in files):raise ValueError('existing study; reconcile first')
    for p,v in files.items():
        with p.open('x') as f:json.dump(v,f,indent=2)
    return str(root/(new+'-flow.local.json'))

if __name__=='__main__':print(prepare(load(Path(sys.argv[1]))))
