"""Complete same-policy eight-seed rewards -> group advantages -> ACE evidence."""
import argparse,collections,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,save,lines,sha,require,canonical,load

def main(config,output):
    cfg=read(OUT/'config/grpo-v1.json')
    for p,h in cfg['reward']['frozen_files'].items():require(sha((ROOT/p).read_bytes())==h,'reward freeze changed')
    scorer=ROOT/'delivery/s-typed-normalized-scoring-v1'
    sys.path.insert(0,str(scorer/'code'));from normalizer import Normalizer
    n=Normalizer(read(scorer/'CONFIG.json'))
    spec=read(config);require(len(spec['members'])==8,'exact8 seed paths required')
    require(sorted(m['seed'] for m in spec['members'])==sorted(cfg['seeds']),'seed group identity')
    gold={r['sample_id']:r for r in lines(scorer/'inputs/baseline545.private.jsonl')}
    sourcecfg=read(ROOT/'delivery/s-correct-p-data-v1/config/materialize-locked.json')
    intents={r['sample_id']:r['gold_OFFLINE_ONLY'] for r in lines(ROOT/sourcecfg['p_ledger']) if r['cohort']=='fit'}
    custody=read(OUT/'config/grpo-policy-custody.private.json')
    require(custody['all_seed_wires_equal_after_seed_removed'] and custody['samples']==545,'actual same policy custody')
    group={};cost=[];policy_id=None
    for member in spec['members']:
        sampling=read(ROOT/member['sampling'])
        if policy_id is None:policy_id=sampling['policy_id']
        require(sampling['policy_id']==policy_id and sampling['seed']==member['seed'],'same policy seed')
        require(sampling['temperature']==cfg['temperature'] and sampling['top_p']==cfg['top_p']
                and sampling['group_reward']==cfg['reward'],'frozen random/reward settings')
        term=read(member['terminal']);require(term['status']=='COMPLETE' and term['unknown']==0,'only known terminal groups')
        cfgpath=ROOT/member['config'];native=read(cfgpath)
        expected=custody['seeds'][str(member['seed'])]
        require(sha(cfgpath.read_bytes())==expected['config_sha256'],'actual config custody')
        require(sha((ROOT/native['manifest']).read_bytes())==native['manifest_sha256']==expected['manifest_sha256'],'actual manifest custody')
        predictions=lines(member['predictions'])
        require(len(predictions)==1635 and len({r['sample_id'] for r in predictions})==1635,'complete unique branch predictions')
        sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'));from aggregate import summarize
        summaries=summarize(lines(ROOT/native['samples']),lines(ROOT/native['manifest']),predictions)
        require(len(summaries)==545,'complete545 per seed')
        for row in summaries:
            sid=row['sample_id'];g=gold[sid];cid=intents[sid]
            require(all(b['status']!='pending' for b in row['branches']),'no pending group branch')
            require(cid is not None,'offline gold intent required')
            branch=next(b for b in row['branches'] if b['candidate_id']==cid)
            valid=branch['status']=='valid';pred=branch['result']['entities'] if valid else []
            def counts(es,typed):
                return collections.Counter((e['type'],n.normalize(e['type'],e['filler']).key if typed else e['filler']) for e in es)
            rewards={}
            for label,typed in [('normalized',True),('strict',False)]:
                gg,pp=counts(g['gold'],typed),counts(pred,typed);tp=sum((gg&pp).values());fp=sum(pp.values())-tp;fn=sum(gg.values())-tp
                reward=(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.) if valid else 0.
                rewards[label]=dict(reward=reward,tp=tp,fp=fp,fn=fn,valid=valid)
            group.setdefault(sid,[]).append(dict(seed=member['seed'],rewards=rewards,
                prediction_sha256=sha(canonical(pred)),prediction=pred,gold=g['gold']))
        cost.append(dict(seed=member['seed'],terminal=member['terminal'],usage=term.get('usage'),wall_seconds=term.get('wall_seconds')))
    output=Path(output);entries=[]
    for sid,members in sorted(group.items()):
        require(len(members)==8,'complete sample group')
        vals=[m['rewards']['normalized']['reward'] for m in members];mean=sum(vals)/8;std=math.sqrt(sum((v-mean)**2 for v in vals)/8)
        for m in members:m['advantage']=(m['rewards']['normalized']['reward']-mean)/std if std else 0.
        entries.append(dict(sample_id=sid,policy_id=policy_id,group_id=policy_id+':'+sid,group_size=8,
            normalized_mean=mean,normalized_std=std,members=members,group_advantage_sum=sum(m['advantage'] for m in members)))
    require(len(entries)==545,'complete545 groups')
    save(output/'group-advantages.private.jsonl',entries,jsonl=True)
    high=sorted(entries,key=lambda r:(-r['normalized_std'],r['sample_id']))[:12]
    save(output/'REFLECTOR-NEXT-INPUT.private.json',dict(schema='s-grpo-ace-feedback-v1',policy_id=policy_id,
        primary_reward=cfg['reward'],samples=545,group_size=8,groups_with_variance=sum(r['normalized_std']>0 for r in entries),
        representative_groups=high,weights_updated=False,curator_action='derive general experience from positive/negative same-policy real trajectories; bounded compiler only',
        cross_policy_pooled=False,costs=cost,official='owner separately computes same original official metric for each full seed; never substitute normalized for official'))
    print(json.dumps(dict(status='ACTUAL_SAME_POLICY_GROUP_ADVANTAGES_COMPLETE',groups=545,group_size=8)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--members',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.members,a.output)
