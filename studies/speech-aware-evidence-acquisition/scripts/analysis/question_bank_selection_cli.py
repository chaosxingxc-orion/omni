"""Prepare cached class similarities, select fixed TRAIN members, and evaluate sealed retrieval."""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
import question_bank_selection as q


def read(p): return json.loads(Path(p).read_bytes())
def save(p, x):
    with Path(p).open('x') as f: json.dump(x, f, ensure_ascii=False, separators=(',', ':')); f.write('\n')


def rank(matrix, members, intents, mask):
    tables={k:np.zeros((matrix.shape[0],len(intents))) for k in ('mean5','coverage','coverage_frozen')}
    for j,intent in enumerate(intents):
        allcols=[i for i,m in enumerate(members) if m['intent']==intent]; cols=[i for i in allcols if mask[i]]
        if not cols: continue
        scores=np.sort(np.maximum(matrix[:,cols],0),axis=1)[:,::-1]; counts=(scores>0).sum(axis=1); n=np.minimum(counts,5)
        mean=np.divide(scores[:,:5].sum(axis=1),n,out=np.zeros(len(n)),where=n>0); s1=scores[:,0]; selected=len({members[i]['sentence_sha256'] for i in cols}); original=len({members[i]['sentence_sha256'] for i in allcols})
        tables['mean5'][:,j]=mean;tables['coverage'][:,j]=mean+.5*(s1-mean)*32/(32+selected);tables['coverage_frozen'][:,j]=mean+.5*(s1-mean)*32/(32+original)
    return {k:[[int(j) for j in np.argsort(-row,kind='stable') if row[j]>0] for row in values] for k,values in tables.items()}


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    for name in ('prepare_features','select','rank','evaluate'):
        x=sub.add_parser(name);x.add_argument('--out',required=True)
        if name in ('prepare_features','select'): x.add_argument('--keys',required=True);x.add_argument('--geometry',required=True)
        if name in ('prepare_features','select','rank'): x.add_argument('--members',required=True);x.add_argument('--rankings',required=True)
        if name=='select': x.add_argument('--cache',required=True);x.add_argument('--source-geometry',required=True);x.add_argument('--config',required=True)
        if name=='rank': x.add_argument('--matrix',required=True);x.add_argument('--matrix-sha256',required=True);x.add_argument('--mask',required=True)
        if name=='evaluate': x.add_argument('--sealed',required=True);x.add_argument('--gold',required=True);x.add_argument('--gold-sha256',required=True);x.add_argument('--baseline',required=True);x.add_argument('--baseline-mean5',required=True);x.add_argument('--classes',required=True)
    a=p.parse_args(); start=time.monotonic()
    if a.mode=='prepare_features': result=q.prepare_cache(a.keys,a.members,a.geometry,a.rankings,a.out)
    elif a.mode=='select': result=q.select(a.cache,a.keys,a.members,a.geometry,a.rankings,a.source_geometry,read(a.config),a.out)
    elif a.mode=='rank':
        if q.digest(a.matrix)!=a.matrix_sha256: raise ValueError('matrix digest differs')
        m=np.load(a.matrix,mmap_mode='r');members=read(a.members);r=read(a.rankings);mask=np.load(a.mask)
        if m.shape!=(len(r['sample_ids']),len(members)) or mask.shape!=(len(members),) or mask.dtype!=bool: raise ValueError('matrix or mask shape differs')
        result={'sample_ids':r['sample_ids'],'intents':r['intents'],'orders':rank(m,members,r['intents'],mask),'gold_read':False,'mask_sha256':q.digest(a.mask),'matrix_sha256':a.matrix_sha256};save(a.out,result)
    else:
        sealed=read(a.sealed)
        if sealed.get('gold_read') is not False or q.digest(a.gold)!=a.gold_sha256: raise ValueError('sealed ranking or gold pin differs')
        gold={x['sample_id']:x['gold_intent'] for x in read(a.gold)}; baseline=read(a.baseline); ids=sealed['sample_ids'];intents=sealed['intents'];idx={v:i for i,v in enumerate(intents)}
        if set(gold)!=set(ids) or baseline['sample_ids']!=ids or baseline['intents']!=intents: raise ValueError('evaluation population differs')
        def ranks(orders):return np.array([row.index(idx[gold[s]])+1 if idx[gold[s]] in row else 0 for s,row in zip(ids,orders)])
        def metric(x):return {'recall1':int(((x>0)&(x<=1)).sum()),'recall3':int(((x>0)&(x<=3)).sum()),'recall5':int(((x>0)&(x<=5)).sum()),'mrr':float(np.divide(1.,x,out=np.zeros(len(x)),where=x>0).mean())}
        baseline_mean=read(a.baseline_mean5)
        if baseline_mean['sample_ids']!=ids or baseline_mean['intents']!=intents: raise ValueError('mean5 population differs')
        b={'mean5':ranks(baseline_mean['orders']['question-gepa-mean5-depth']),'coverage':ranks(baseline['orders']['question-gepa-coverage-gated-depth'])}
        classes=read(a.classes)
        result={}
        for name,orders in sealed['orders'].items():
            x=ranks(orders);comp=b['mean5' if name=='mean5' else 'coverage']; paired={}
            if comp is not None:
                for k in (1,3,5):
                    aa=(comp>0)&(comp<=k);bb=(x>0)&(x<=k);paired[str(k)]={'repair':int((~aa&bb).sum()),'regression':int((aa&~bb).sum())}
            sizes={t:classes[t]['original'] for t in intents}; strata={}
            for label, predicate in [('1-32',lambda n:0<n<=32),('33-128',lambda n:32<n<=128),('129+',lambda n:n>128)]:
                selected=np.array([predicate(sizes[gold[s]]) for s in ids]); strata[label]={'samples':int(selected.sum()),**metric(x[selected])} if selected.any() else {'samples':0}
            result[name]={'metrics':metric(x),'paired':paired,'original_class_size_strata':strata,'fewer_than3':sum(len(v)<3 for v in orders),'fewer_than5':sum(len(v)<5 for v in orders),'per_intent':{t:metric(x[np.array([gold[s]==t for s in ids])]) for t in intents if any(gold[s]==t for s in ids)}}
        save(a.out,{'results':result,'population':len(ids),'development_only':True,'gold_sha256':a.gold_sha256,'sealed_sha256':q.digest(a.sealed),'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(json.dumps({'mode':a.mode,'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},sort_keys=True))
if __name__=='__main__':main()
