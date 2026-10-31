"""Audit real construction625 retrieval diversity without inference or gold.

Shared provider ranking is authoritative. Fused IDs are sorted identifiers, not
rank order; never slice their positions as retrieval ranks. No runtime mutation.
"""
import argparse
import copy
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/flights/nx')]
from fold_a_retrieval_inputs import read,sha,raw,confined
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
STATE=None


def ordered_values(entries,values,selected):
    scores={e.entry_id:e.score for e in entries}
    by_id={v.value_id:v for v in values}
    if len(by_id)!=len(values) or len(set(selected))!=len(selected):raise ValueError('duplicate values')
    result=[]
    for value_id in selected:
        value=by_id[value_id]
        support=[scores[f] for f in value.associated_facet_ids if f in scores]
        if not support:raise ValueError('selected value has no ranked supporting facet')
        result.append((max(support),value_id))
    return [v for _,v in sorted(result,key=lambda x:(-x[0],x[1]))]


def variant(original,wide):
    if len(set(original))!=len(original) or len(set(wide))!=len(wide):raise ValueError('duplicate evidence')
    if wide[:len(original)]!=original:raise ValueError('top40 does not preserve original rank prefix')
    fallback=len(original)!=20 or len(wide)<40
    return list(original if fallback else wide[:40:2]),fallback


def initialize(providers,queries,packets,split,values):
    global STATE
    STATE=(providers,queries,packets,split,values)


def audit_one(sample_id):
    from speech_aware_evidence_acquisition.rails.phase_supply import QueryState
    providers,queries,packets,split,values=STATE
    descriptor=packets[sample_id]
    state=QueryState(query_id='seed-phase:solver:'+sample_id,query_sample_id=sample_id,
        speech_packet_id=descriptor['packet_id'],speech_packet_sha256=descriptor['sha256'],
        use_split_identity_hash=split[sample_id],query_source='speech_features').validate()
    ranked=[];fused=[]
    for k,provider in zip((20,40),providers):
        ranking=provider.candidates({'query_state':state,'query_vectors':dict(queries[sample_id])},k=k)
        selected=list(provider.fuse([ranking],k=k,policy_id='saea-fuse-supplyfilter-v1').selected_value_ids)
        fused.append(selected);ranked.append(ordered_values(ranking.entries,values,selected))
    alternative,fallback=variant(*ranked)
    if len(alternative)>20:raise ValueError('candidate evidence bound exceeded')
    if len(providers)==3:
        provider=providers[2]
        ranking=provider.candidates({'query_state':state,'query_vectors':dict(queries[sample_id])},k=40)
        actual=provider.fuse([ranking],k=40,policy_id='saea-fuse-supplyfilter-odd40-v1')
        if list(actual.selected_value_ids)!=sorted(alternative):
            raise ValueError('shared odd40 supply differs from audited candidate')
    return {'sample_id':sample_id,'original_render_ids':fused[0],'original_rank_ids':ranked[0],
        'wide_rank_ids':ranked[1],'alternative_render_ids':sorted(alternative),'fallback':fallback,
        'changed':set(alternative)!=set(fused[0]),'overlap':len(set(alternative)&set(fused[0]))}


def run(repo,source,data,out,verify_odd40=False):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.phase_supply import load_phase_supply_specs,build_supply_for_phase
    from build_slurp_intent_exemplar_agreement import load_snapshot,utterance_identity
    assert_execution_environment();head=require_clean_study_repo(repo);source_head=require_clean_study_repo(source)
    out=confined(data,str(out))
    if out.exists() or not out.is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned audit output required')
    providers=None;packets={};ids=[];pins={};halves={};original_store=None;split={}
    for half in ('h1','h2'):
        name=f'slurp-tfrl-current-seed-i1-{half}-v1';ep=f'configs/experiment/{name}.json';bp=f'configs/baseline/{name}.json'
        exp=read(source/ep);base=read(source/bp);cell,=exp['cells'];roster=exp['populations'][0]['ids']
        if len(roster)!=(313 if half=='h1' else 312) or set(roster)&set(ids):raise ValueError('exact disjoint halves required')
        viewrel=f"configs/dataset/{cell['dataset_fragment']}.json";view=read(source/viewrel)
        if view['count']!=625 or len(view['items'])!=625:raise ValueError('full625 real dataset required')
        for row in view['items']:
            if not row['sample_id'].startswith('slurp:train:') or '/slurp_real/' not in row['media_relpath'] or not row['audio_sha256']:
                raise ValueError('real audio dataset required')
        store=base['knowledge_stores']['train-exemplars']
        if original_store is not None and store!=original_store:raise ValueError('half retrieval differs')
        original_store=store
        for rel in (ep,bp,viewrel):pins[rel]=sha(source/rel)
        runtime=build_flight_runtime(source,data,model=cell['model_fragment'],dataset=cell['dataset_fragment'],
            baseline=cell['baseline_fragment'],experiment=name,run_id=cell['run_id'],attempt_id=cell['attempt_id'],
            runtime_root=data/'work/scratch/runtime-root',lock_path=source.parents[1]/'docs/datasets.lock.json')
        _,binding,_,_,actual_split,_=runtime.phase_supply_artifact_identity()
        if actual_split!=exp['population']['split_identity_hash']:raise ValueError('runtime split differs')
        split.update({row_id:actual_split for row_id in roster})
        own=dict(binding['speech_packet_bindings'])
        if set(own)!=set(roster):raise ValueError('runtime real roster differs')
        packets.update(own);ids+=roster;halves[half]=list(roster)
    if len(ids)!=625 or len(set(ids))!=625:raise ValueError('complete625 required')
    snapshot,snapshot_sha=load_snapshot(data,original_store['snapshot'])
    source_utterances={utterance_identity(s) for v in snapshot.values for s in v.source_sample_ids}
    if {utterance_identity(s) for s in ids}&source_utterances:raise ValueError('query utterances overlap evidence pool')
    phase=original_store['phase_supply']['solver']
    if (phase['k']!=20 or phase['fuse_coverage']!={'m':20,'c':20,'top_l':20}
            or phase['rank_weights']!={'w_cos':1.,'w_prior':0.,'w_agree':0.}):raise ValueError('original top20 baseline required')
    providers=[];query_table=None;data_pins={}
    for k in ((20,40,40) if verify_odd40 else (20,40)):
        config=copy.deepcopy(original_store['phase_supply']);config['solver']['k']=k
        config['solver']['fuse_coverage']={'m':k,'c':k,'top_l':k}
        if len(providers)==2:config['solver']['fuse_policy']='saea-fuse-supplyfilter-odd40-v1'
        specs=load_phase_supply_specs({'phase_supply':config},default_phase_program('phase-native-module-v1-c7'))
        callback=build_supply_for_phase(specs,snapshot=snapshot,snapshot_sha256=snapshot_sha,data_root=data,seed_phases=('solver',))
        queries=callback.query_tables['solver']
        if query_table is not None and queries!=query_table:raise ValueError('query vectors changed between widths')
        query_table=queries;providers.append(callback.provider_factories['solver']())
    if set(ids)-set(query_table):raise ValueError('missing real query vectors')
    for key in ('query_source','facet_source'):
        pin=phase[key];path=confined(data,pin['relpath'])
        if sha(path)!=pin['sha256']:raise ValueError('vector pin changed')
        data_pins[pin['relpath']]=sha(path)
    out.mkdir(parents=True)
    publish_new(out/'intent.json',raw({'schema':'saea-tfrl-distinct-evidence-audit-intent-v1','source_commit':source_head,
        'implementation_commit':head,'source_hashes':pins,'data_hashes':data_pins,'rows':625,'new_model_calls':0,
        'shared_odd40_verification':verify_odd40,'query_gold_read':False,'candidate_rule':'original rank1..20 vs odd ranks1..39 from top40; retain original if fewer than40',
        'preparation_gate':'strict majority of625 rows have changed supply; otherwise stop evidence intervention preparation'}))
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('fork'),initializer=initialize,
            initargs=(providers,query_table,packets,split,snapshot.values)) as pool:
        results=list(pool.map(audit_one,ids,chunksize=8))
    if require_clean_study_repo(repo)!=head or require_clean_study_repo(source)!=source_head:raise ValueError('source changed during audit')
    for rel,digest in pins.items():
        if sha(source/rel)!=digest:raise ValueError('source input changed')
    for rel,digest in data_pins.items():
        if sha(data/rel)!=digest:raise ValueError('vector input changed')
    publish_new(out/'rows.jsonl',b''.join(raw(r).replace(b'\n',b'')+b'\n' for r in results))
    changed=sum(r['changed'] for r in results);fallback=sum(r['fallback'] for r in results)
    result={'schema':'saea-tfrl-distinct-evidence-audit-v1','rows':625,'source_commit':source_head,'implementation_commit':head,
        'changed_rows':changed,'fallback_rows':fallback,'mean_overlap':sum(r['overlap'] for r in results)/625,
        'pool_utterance_overlap':0,'snapshot_sha256':snapshot_sha,'source_hashes':pins,'data_hashes':data_pins,
        'row_artifact_sha256':sha(out/'rows.jsonl'),'new_model_calls':0,'tokenizer_calls':0,'scoring_jobs':0,
        'shared_odd40_verified':verify_odd40,'query_gold_read':False,'preparation_gate_passed':changed>312,'generation_authorized':False,
        'scope':'Model-free real query ranking diversity; no inference performance or quality claim.',
        'remaining':['delivery rule and cross-fit adaptation','actual candidate runtime integration','FIT budgets','prospective generation registration']}
    publish_new(out/'complete.json',raw(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','source','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--verify-odd40',action='store_true')
    print(json.dumps(run(**vars(p.parse_args()))))
