"""Measure actual contiguous sampled windows from the original control-host journal.

This report verifies timings. It is not a release or a numerical acceptance.
"""
import argparse,hashlib,json,math
from pathlib import Path


def read(p):
    return json.loads(p.read_bytes())


def authenticated(path,ref):
    data=path.read_bytes()
    if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']:
        raise ValueError('optimizer evidence digest differs: '+str(path))
    return json.loads(data)


def observed_updates(root):
    """A returned transport envelope alone does not prove an optimizer step."""
    updates={}
    for folder in sorted(root.glob('step-*/update/returned')):
        commit=folder/'CLOSED-COMMIT.json'
        if not commit.is_file():continue
        manifest=folder/'CLOSED-MANIFEST.json'; marker=read(commit)
        if not marker['immutable'] or hashlib.sha256(manifest.read_bytes()).hexdigest()!=marker['manifest_sha256']:
            raise ValueError('optimizer return manifest differs')
        entries=read(manifest);ret=authenticated(folder/'RETURN.json',entries['RETURN.json'])
        if ret['optimizer_steps']==0:continue
        locations=read(folder/'LOCATION-MAP.json')
        def member(ref):
            path=Path(locations[ref['path']]).resolve()
            if not path.is_relative_to(folder.resolve()) or entries[str(path.relative_to(folder.resolve()))]!=ref:
                raise ValueError('optimizer observation outside closed return')
            return authenticated(path,ref)
        result=member(ret['result_ref']);point=member(ret['checkpoint_ref'])
        obj=result['result']['objective_observation'];step=point['step']
        if (result['status']!='updated' or ret['optimizer_steps']!=1 or result['optimizer_steps']!=1 or
                not result['formal_rl_enabled'] or not point['recoverable'] or
                result['checkpoint_ref']!=ret['checkpoint_ref'] or result['clock']!=point['clock'] or
                obj['successful_step_after']!=step or obj['successful_step_before']!=step-1 or
                point['clock']['successful_optimizer_steps']!=step or not obj['optimizer_step'] or
                not math.isfinite(obj['components']['total']) or not math.isfinite(obj['gradient_norm']) or
                obj['gradient_norm']<=0):
            raise ValueError('missing finite, committed formal update')
        selected=member(result['selected_cohort_ref'])
        ranks=[member(ref) for ref in result['rank_refs']]
        if (point['world_size']!=8 or sorted(r['rank'] for r in ranks)!=list(range(8)) or
                any(r['step']!=step or not r['recoverable_checkpoint'] or r['objective']['components']!=obj['components'] for r in ranks) or
                result['group_ids']!=selected['group_ids'] or point['training_lineage']['group_ids']!=selected['group_ids']):
            raise ValueError('rank/group evidence differs from committed step')
        key=result['generation_inventory_ref']['sha256']
        if key in updates:raise ValueError('duplicate optimizer return for the same rollout')
        updates[key]=(commit,step,obj,result['selected_cohort_ref']['sha256'],ret['checkpoint_ref']['sha256'])
    return updates


def report(root,policy,after):
    limits=read(policy)['acceptance']; windows=[]
    anchor=None;updates=observed_updates(root)
    for c in sorted((root/'rollout').glob('*-capture')):
        cohort=c/'cohort'; closed=cohort/'GENERATION-CLOSED.json'
        if not closed.is_file():continue
        selected=read(cohort/'SELECTED.json')
        step=read(Path(selected['policy']['current_ref']['path']))['step']+1
        if step<=after:continue
        n,g=limits['recordings_per_step'],limits['episodes_per_recording']
        if len(selected['draws'])!=n or len(read(closed)['groups'])!=n:raise ValueError('incomplete cohort')
        records=set(); rosters=set();dispatch=[];responses=[]
        for draw in selected['draws']:
            group=cohort/'groups'/str(draw['draw_index'])
            generated=read(group/'GENERATED.json');rec=draw['record']['recording_ref_id']
            if rec in records or len(generated['episodes'])!=g:raise ValueError('recording/G differs')
            records.add(rec)
            for ep in generated['episodes']:
                call,=ep['calls']
                req=call['packet']['request_id']
                if req in rosters:raise ValueError('duplicate sampled call')
                rosters.add(req)
                folder=group/f"episode-{ep['replicate']}"/'native-journal'
                d,=folder.glob('*.dispatch.json');r,=folder.glob('*.response.json')
                sent,returned=read(d),read(r)
                if (sent['request_id']!=req or returned['request_id']!=req or
                        sent['request_sha256']!=returned['request_sha256'] or sent['intent_sha256']!=returned['intent_sha256'] or
                        d.name.split('.')[0]!=r.name.split('.')[0] or r.stat().st_mtime_ns<d.stat().st_mtime_ns):
                    raise ValueError('journal response timing/identity mismatch')
                dispatch.append(d.stat().st_mtime_ns);responses.append(r.stat().st_mtime_ns)
        first=min(dispatch)
        if anchor:
            previous,start,last,inventory,selection=anchor
            if step!=previous+1 or first<=last:raise ValueError('non-contiguous sampled updates')
            if inventory not in updates:raise ValueError('no formal update for this exact rollout')
            commit,updated,obj,training_selection,checkpoint=updates[inventory]
            if (updated!=previous or training_selection!=selection or obj['global_episode_denominator']!=n*g or
                    obj['groups']!=n):raise ValueError('training update does not match measured roster')
            if not read(commit)['immutable'] or not start<commit.stat().st_mtime_ns<first:
                raise ValueError('missing intervening optimizer return')
            seconds=(first-start)/1e9
            windows.append(dict(step=previous,recordings=n,episodes=n*g,seconds=seconds,
                within_limit=seconds<=limits['maximum_step_seconds'],first_dispatch_ns=start,
                successor_first_dispatch_ns=first,loss=obj['components']['total'],checkpoint_sha256=checkpoint,commit_sha256=hashlib.sha256(commit.read_bytes()).hexdigest()))
        anchor=(step,first,max(responses),hashlib.sha256((cohort/'GENERATION-INVENTORY.json').read_bytes()).hexdigest(),
                hashlib.sha256((cohort/'SELECTED.json').read_bytes()).hexdigest())
    k=limits['consecutive_steps']
    return dict(scope='timing_only',timing=limits['timing'],steps=windows,maximum_seconds=limits['maximum_step_seconds'],
        latest_consecutive_timing_pass=len(windows)>=k and all(w['within_limit'] for w in windows[-k:]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--after',type=int,required=True)
    a=p.parse_args();print(json.dumps(report(a.run,a.config,a.after),indent=2))
