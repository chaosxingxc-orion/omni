"""Prepare held Fold A agreement inputs from the completed two-contact A0.

Uses the adopted CPU encoding, preserves the original disjoint exemplar pool,
and disables only circular agreement. This entry does not admit generation.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import fold_a_adopted_retrieval_inputs as inputs
from fold_a_current_comparator import CELL,RUN
from fold_a_retrieval_inputs import read,sha,raw,confined
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from r01c_pool_audit import utterance


def verify_membership(ids,snapshot):
    if len(ids)!=1080 or len(set(ids))!=1080:raise ValueError('unique complete Fold A required')
    query={utterance(s) for s in ids};source=set(snapshot.source_sample_ids)
    if not source or len(source)!=len(snapshot.source_sample_ids) or query & {utterance(s) for s in source}:
        raise ValueError('pool overlaps Fold A or has invalid membership')
    facets={f.facet_id for f in snapshot.facets}
    if len(facets)!=len(snapshot.facets):raise ValueError('duplicate facets')
    for obj in (*snapshot.facets,*snapshot.values):
        if not obj.source_sample_ids or not set(obj.source_sample_ids)<=source:
            raise ValueError('hidden pool membership')
    for value in snapshot.values:
        if not value.task_target or not set(value.associated_facet_ids)<=facets:
            raise ValueError('missing vote target or facet')
    return {'query_rows':1080,'query_utterances':len(query),'pool_source_recordings':len(source),
        'pool_facets':len(facets),'pool_values':len(snapshot.values),'utterance_overlap':0}


def derive(repo,baseline,experiment,pins):
    b,e=copy.deepcopy(baseline),copy.deepcopy(experiment)
    block=b['native_episode']['consumption_module']
    cm.verify_declared_module(block,b,repo_root=repo)
    if b['policy']!='phase-native-module-v1-c2' or b['native_episode']['review_loop']!='off' or block['lock']!='v2.2-npu-port':
        raise ValueError('original A0 two-contact port method required')
    roster=read(repo/inputs.ROSTER);ids=[r['sample_id'] for r in roster['items']]
    if len(ids)!=1080 or len(set(ids))!=1080 or any(not s.startswith('slurp:train:') for s in ids):
        raise ValueError('complete Fold A train roster required')
    for name in (inputs.SEGMENTS,inputs.CONTENT,inputs.RECEIPT,inputs.ADOPTION,inputs.ALIAS_RECEIPT,inputs.PROVENANCE):
        v=pins.get(name)
        if not isinstance(v,str) or len(v)!=64 or set(v)-set('0123456789abcdef') or v=='0'*64:
            raise ValueError('real adopted input pin required')
    for store,phase,rel in [('train-exemplars','solver',inputs.CONTENT),('intent-boundaries','perception',inputs.SEGMENTS)]:
        q=b['knowledge_stores'][store]['phase_supply'][phase]['query_source']
        q.update(relpath=rel,sha256=pins[rel],tool_contact_ledger_row=inputs.RUN)
        q['embedded_signal']=q['embedded_signal'].replace('Fold B','Fold A')
    for i,c in enumerate(block['components']):
        if c['component_id']=='agree-vote-seed':block['components'][i]={'component_id':'agree-vote-seed','state':'off'}
    compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    target=b['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    expected=compiled.values_on_roster_of(b)['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    target.pop('agreement_source',None)
    for k in ('k','rank_weights'):target[k]=expected[k]
    b.pop('prepared_phase_supply',None)
    b['preparation_source_receipts']=[p for p in b['preparation_source_receipts'] if 'FOLD-B' not in p['relpath'] and 'fold-b' not in p['relpath']]
    b['preparation_source_receipts'] += [{'root':'data','relpath':p,'sha256':pins[p]} for p in
        (inputs.RECEIPT,inputs.ADOPTION,inputs.ALIAS_RECEIPT,inputs.PROVENANCE)]
    b['preparation_source_receipts'].append({'root':'repository','relpath':inputs.ROSTER,'sha256':sha(repo/inputs.ROSTER)})
    b['native_contact_fit'].update(run_id=RUN+'-FIT',attempt_id='never-flown-fit')
    b['what_this_is']='Held current c2 Fold A agreement bootstrap; original pool and recall preserved.'
    cm.verify_declared_module(block,b,repo_root=repo)
    if 'agreement_source' in json.dumps(b):raise ValueError('circular agreement remains')
    cell,=e['cells'];cell.update(cell_id=CELL,baseline_fragment=CELL,dataset_fragment=Path(inputs.DATASET).stem,
        run_id=RUN,attempt_id='never-flown',allowed_execution_stages=['check'],
        readiness={'state':'held','requirements':[],'reason':'Model-free agreement input; no generation admission.'})
    e.pop('baseline_predecessor',None)
    e.update(experiment_id=RUN,split_id=read(repo/inputs.DATASET)['split_name'],purpose='Current c2 model-free Fold A agreement bootstrap.',
        population={'count':1080,'dataset_fragment':Path(inputs.DATASET).stem,'roster_receipt':inputs.ROSTER,
            'split_identity_hash':roster['split_identity_hash']},populations=[{'dataset_fragment':Path(inputs.DATASET).stem,'ids':ids}])
    e['stage3_measurement'].update(rows=1080,read_roster=roster['roster_id'],claims_benefit=False)
    return b,e


def write(repo,source_repo,encoder_repo,data,result_path,retrieval_complete,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.rails.knowledge_stores import load_knowledge_store_binding
    from speech_aware_evidence_acquisition.rails.phase_supply import load_sealed_vectors
    assert_execution_environment();head=require_clean_study_repo(repo);source_head=require_clean_study_repo(source_repo)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output required')
    result=read(result_path)
    if (result.get('schema')!='saea-real500-verified-result-v1' or result.get('rows')!=500
            or not result.get('official_finalized') or not result.get('checkpoint_cost_reconciled')):
        raise ValueError('completed real500 source required')
    pin=result['selection_evidence']['manifest'];mp=confined(data,pin['relpath'])
    if sha(mp)!=pin['sha256']:raise ValueError('manifest changed')
    bundle=verify_bundle(mp);f=bundle.manifest['config']['fragments']
    if (bundle.attempt_state!='FINALIZED' or len(bundle.outputs_rows)!=500 or bundle.manifest['study_commit']!=source_head
            or bundle.manifest['run_id']!=result['run_id'] or compose(source_repo,**f).config_hash!=bundle.manifest['config']['config_hash']):
        raise ValueError('completed source identity changed')
    prep=read(retrieval_complete)
    if (prep.get('schema')!='saea-fold-a-adopted-retrieval-inputs-v1' or prep.get('rows')!=1080
            or prep.get('values_unchanged') is not True or prep.get('encoder_run_id')!=inputs.RUN):
        raise ValueError('adopted Fold A preparation required')
    for rel,digest in prep['pins'].items():
        if sha(confined(data,rel))!=digest:raise ValueError('retrieval artifact changed')
    alias,_=inputs.source_config(encoder_repo,data)
    if sha(repo/inputs.ROSTER)!=sha(encoder_repo/inputs.ROSTER) or sha(repo/inputs.DATASET)!=sha(encoder_repo/inputs.DATASET):
        raise ValueError('encoder roster or dataset differs')
    base=read(source_repo/f"configs/baseline/{f['baseline']}.json")
    binding=load_knowledge_store_binding(base,data_root=data);snapshot=binding.stores['train-exemplars'].snapshot
    membership=verify_membership([r['sample_id'] for r in read(repo/inputs.ROSTER)['items']],snapshot)
    facet=base['knowledge_stores']['train-exemplars']['phase_supply']['solver']['facet_source']
    vectors=load_sealed_vectors(confined(data,facet['relpath']),expected_sha256=facet['sha256'],dimensions=('content',))
    if any(f.facet_id not in vectors or len(vectors[f.facet_id]['content'])!=alias['vector_width'] for f in snapshot.facets):
        raise ValueError('incomplete exemplar vector coverage')
    b,e=derive(repo,base,read(source_repo/f"configs/experiment/{f['experiment']}.json"),prep['pins'])
    blobs={f'configs/baseline/{CELL}.json':raw(b),f'configs/experiment/{CELL}.json':raw(e)}
    # The shared agreement builder expects exactly two output config pins.
    # Models are staged independently, verified byte-identical, before this call.
    modelrel=f"configs/model/{f['model']}.json";model=read(source_repo/modelrel)
    for rel in (modelrel,'docs/'+model['runtime_receipt']):
        if (repo/rel).read_bytes()!=(source_repo/rel).read_bytes():raise ValueError('stage exact model inputs first')
    if any((repo/rel).exists() for rel in blobs) or any((data/k/RUN).exists() for k in ('runs','attempts','prepared')):
        raise ValueError('comparator identity already used')
    record={'schema':'saea-fold-a-c2-comparator-v1','source_commit':source_head,'implementation_commit':head,'rows':1080,
        'source_manifest_sha256':sha(mp),'retrieval_complete_sha256':sha(retrieval_complete),'pool_membership':membership,
        'pool_snapshot_sha256':snapshot.content_hash(),'model_contacts':0,'allowed_execution_stages':['check'],
        'claims_system_lock':False,'output_hashes':{p:hashlib.sha256(v).hexdigest() for p,v in blobs.items()}}
    publish_new(out/'intent.json',raw(record))
    for rel,blob in blobs.items():publish_new(repo/rel,blob);(repo/rel).chmod(0o644)
    publish_new(out/'complete.json',raw(record));return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','source-repo','encoder-repo','data','result-path','retrieval-complete','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(write(**vars(p.parse_args()))))
