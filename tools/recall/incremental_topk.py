"""Exact fixed-K insertion delta; source masking precedes TopK and legal labels follow it."""
import time
import numpy as np

class IncrementalTopK:
 def __init__(self,S,y,groups,rows,legal,dr,gold,k,d):
  self.S=S;self.y=np.asarray(y);self.groups=np.asarray(groups);self.rows=rows
  self.legal=legal;self.dr=dr;self.gold=gold;self.k=k;self.d=d
  _,self.gid=np.unique(self.groups,return_inverse=True)
  self.excluded={g:np.array([g in r['excluded_sourcegroups'] for r in rows]) for g in set(groups)}
  self.n=len(rows);self.L=legal.shape[1];self.ii=np.arange(self.n)
 def labels_for(self,ix):
  valid=ix>=0; yy=self.y[np.maximum(ix,0)]
  counts=np.zeros((len(ix),self.L),np.int16)
  rr=np.broadcast_to(np.arange(len(ix))[:,None],ix.shape)
  np.add.at(counts,(rr[valid],yy[valid]),1)
  return counts>0
 def reset(self,orders):
  self.top=np.full((self.n,self.k+1),-1,np.int32)
  for i,o in enumerate(orders):self.top[i,:min(len(o),self.k+1)]=o[:self.k+1]
  self.q=self.labels_for(self.top[:,:self.k])&self.legal
  self.hit=self.q[self.ii,self.gold];self.union=self.hit|(self.legal[self.ii,self.gold]&(self.dr[self.ii,self.gold]<=self.d))
 def prefixes(self,j):
  # Only an eligible key outranking the K+1 boundary can change the retained prefix.
  last=self.top[:,-1]; safe=np.maximum(last,0); sj=self.S[:,j]; sb=self.S[self.ii,safe]
  change=~self.excluded[self.groups[j]]&((last<0)|(sj>sb)|((sj==sb)&(j<last)))
  ids=np.flatnonzero(change)
  if not len(ids):return ids,self.top[:0]
  old=self.top[ids];valid=old>=0;safe=np.maximum(old,0);scores=self.S[ids[:,None],safe]
  same=valid&(self.gid[safe]==self.gid[j])
  better_old=same&((scores>sj[ids,None])|((scores==sj[ids,None])&(old<=j)))
  keep=~better_old.any(axis=1);ids=ids[keep];old=old[keep];same=same[keep]
  a=np.concatenate((np.where(same,-1,old),np.full((len(ids),1),j,np.int32)),axis=1)
  score=self.S[ids[:,None],np.maximum(a,0)];score=np.where(a>=0,score,-np.inf)
  order=np.lexsort((np.where(a>=0,a,np.iinfo(np.int32).max),-score),axis=1)
  new=np.take_along_axis(a,order,axis=1)[:,:self.k+1]
  return ids,new
 def trial(self,j):
  ids,top=self.prefixes(j);q=self.labels_for(top[:,:self.k])&self.legal[ids]
  hit=q[np.arange(len(ids)),self.gold[ids]]
  union=hit|(self.legal[ids,self.gold[ids]]&(self.dr[ids,self.gold[ids]]<=self.d))
  qg=~self.hit[ids]&hit;ql=self.hit[ids]&~hit
  ug=~self.union[ids]&union;ul=self.union[ids]&~union
  repairs=qg if self.primary_metric=='question' else ug
  ri=np.flatnonzero(repairs);margins=[]
  for a in ri:
   p=top[a];key=next(int(v) for v in p[:self.k] if v>=0 and self.y[v]==self.gold[ids[a]])
   boundary=float(self.S[ids[a],p[self.k]]) if p[self.k]>=0 else -1.
   margins.append(float(self.S[ids[a],key])-boundary)
  return dict(question_gains=int(qg.sum()),question_losses=int(ql.sum()),union_gains=int(ug.sum()),union_losses=int(ul.sum()),label_lost=int((self.q[ids]&~q).sum()),label_new=int((~self.q[ids]&q).sum()),repair_margins=np.array(margins),affected_queries=len(ids))
 primary_metric='question'
 def verify_reference(self,selected,clean,ex,nq,nc):
  t=time.monotonic();candidates=sorted(set(clean)-set(selected))[:nc]
  # Spread through all roles and include candidates sharing selected source groups.
  shared=[j for j in clean if j not in selected and self.groups[j] in set(self.groups[selected])]
  candidates=list(dict.fromkeys(candidates+shared[:2]))
  queries=np.unique(np.linspace(0,self.n-1,nq,dtype=int));checked=0
  for j in candidates:
   ids,top=self.prefixes(j);lookup={int(i):a for a,i in enumerate(ids)}
   for i in queries:
    old=ex.source_order(self.S[i],selected,self.groups,self.rows[i]['excluded_sourcegroups'])
    new=ex.source_order(self.S[i],selected+[j],self.groups,self.rows[i]['excluded_sourcegroups'])
    expected=new[:self.k+1];got=top[lookup[int(i)]].tolist() if int(i) in lookup else self.top[i].tolist();got=[v for v in got if v>=0]
    assert got==expected,(int(i),j,'prefix mismatch')
    _,ql=ex.retrieve(self.S[i],new,self.y,self.legal[i],dict(mode='source_topk',source_budget=self.k))
    actual=np.flatnonzero(self.labels_for(np.array([got[:self.k]+[-1]*max(0,self.k-len(got))]))[0]&self.legal[i]).tolist()
    assert sorted(ql)==actual
    checked+=1
  # Explicit ties, source aliases, excluded aliases, and illegal labels.
  fixture_score=np.array([[.8,.8,.9,.7,.6,.6],[.4,.4,.3,.8,.8,.1]])
  fixture_groups=['a','b','a','c','d','e'];fy=np.array([0,1,0,2,1,2]);fr=[dict(excluded_sourcegroups=['c']),dict(excluded_sourcegroups=['a'])]
  fl=np.array([[True,False,True],[True,True,True]]);fg=np.array([0,2]);fd=np.full((2,3),9)
  e=IncrementalTopK(fixture_score,fy,fixture_groups,fr,fl,fd,fg,2,5);sel=[0,1,3,4]
  e.reset([ex.source_order(fixture_score[i],sel,fixture_groups,fr[i]['excluded_sourcegroups']) for i in range(2)])
  for j in [2,5]:
   ids,p=e.prefixes(j);lookup={int(i):a for a,i in enumerate(ids)}
   for i in range(2):
    want=ex.source_order(fixture_score[i],sel+[j],fixture_groups,fr[i]['excluded_sourcegroups'])[:3]
    got=p[lookup[i]].tolist() if i in lookup else e.top[i].tolist()
    assert [v for v in got if v>=0]==want
  return dict(status='PASS',real_query_candidate_pairs=checked,synthetic_alias_mask_tie_pairs=4,reference='unchanged original source_order/retrieve',prefix_depth=self.k+1,wall_seconds=time.monotonic()-t)
