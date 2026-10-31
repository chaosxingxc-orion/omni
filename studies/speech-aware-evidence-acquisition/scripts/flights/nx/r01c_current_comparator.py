"""Derive a held R01C agreement input from the completed real500 A0 method.

No generation or system-gain claim. Preserve actual two-contact semantics and
recall, substitute the disjoint pool and current sealed R01C speech vectors, and
disable only agreement for the model-free agreement bootstrap.
"""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import json

REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
import r01c_retrieval_inputs as inputs
from fold_a_retrieval_inputs import read,sha,raw,confined
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_lock_v2_2_npu_port_r01c import (
    LOCK_V2_2_NPU_PORT_R01C, R01C_RECEIPT_RELPATH, r01c_pins, POOL_PATH)

CELL='slurp-a0-r01c-current-agreement-comparator-v1'
RUN='SAEA-SLURP-A0-R01C-CURRENT-AGREEMENT-COMPARATOR-V1'
PKG='docs/readiness/2026-10-31-r01c-current-comparator-v1'


def derive(repo,baseline,experiment,pins):
    baseline,experiment=copy.deepcopy(baseline),copy.deepcopy(experiment)
    cm.verify_declared_module(baseline['native_episode']['consumption_module'],baseline,repo_root=repo)
    if (baseline['policy']!='phase-native-module-v1-c2'
            or baseline['native_episode'].get('review_loop')!='off'):
        raise ValueError('current two-contact A0 source required')
    roster=read(repo/inputs.ROSTER);ids=[r['sample_id'] for r in roster['items']]
    if len(ids)!=625 or len(set(ids))!=625 or any(not s.startswith('slurp:train:') for s in ids):
        raise ValueError('complete R01C625 train roster required')
    for name in (inputs.SEGMENTS,inputs.CONTENT,inputs.RECEIPT,inputs.ALIAS_RECEIPT,inputs.PROVENANCE):
        value=pins.get(name)
        if not isinstance(value,str) or len(value)!=64 or set(value)-set('0123456789abcdef') or value=='0'*64:
            raise ValueError('missing real R01C input pin')
    stores=baseline['knowledge_stores'];pool=stores['train-exemplars']
    store_pins=r01c_pins(repo)
    for side in ('release','snapshot'):
        pool[side]={key:store_pins[POOL_PATH+(side,key)] for key in ('relpath','sha256')}
    for store,phase,name in [('train-exemplars','solver',inputs.CONTENT),
                              ('intent-boundaries','perception',inputs.SEGMENTS)]:
        query=stores[store]['phase_supply'][phase]['query_source']
        query.update(relpath=name,sha256=pins[name],tool_contact_ledger_row=inputs.RUN)
        query['embedded_signal']=query['embedded_signal'].replace('TRAIN Fold B','TRAIN R01C')
    recall=copy.deepcopy(pool['phase_supply']['solver'])
    block=baseline['native_episode']['consumption_module']
    block['lock']=LOCK_V2_2_NPU_PORT_R01C
    for i,component in enumerate(block['components']):
        if component['component_id']=='agree-vote-seed':
            block['components'][i]={'component_id':'agree-vote-seed','state':'off'}
    compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    perception=stores['intent-boundaries']['phase_supply']['perception']
    expected=compiled.values_on_roster_of(baseline)['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    perception.pop('agreement_source',None)
    for key in ('k','rank_weights'):perception[key]=expected[key]
    baseline.pop('prepared_phase_supply',None)
    baseline['preparation_source_receipts']=[p for p in baseline['preparation_source_receipts']
        if not any(part in p['relpath'] for part in ('FOLD-B','fold-b','slurp-kb-size-pool-annotated-exemplars.json'))]
    baseline['preparation_source_receipts'] += [
        {'root':'data','relpath':name,'sha256':pins[name]} for name in
        (inputs.RECEIPT,inputs.ALIAS_RECEIPT,inputs.PROVENANCE)]
    baseline['preparation_source_receipts'] += [
        {'root':'repository','relpath':name,'sha256':sha(repo/name)} for name in (inputs.ROSTER,R01C_RECEIPT_RELPATH)]
    baseline['native_contact_fit'].update(run_id=RUN+'-FIT',attempt_id='never-flown-fit')
    baseline['what_this_is']='Held model-free R01C agreement bootstrap from the actual two-contact A0 method.'
    cm.verify_declared_module(block,baseline,repo_root=repo)
    if pool['phase_supply']['solver']!=recall or 'agreement_source' in json.dumps(baseline):
        raise ValueError('bootstrap altered recall or retained circular agreement')
    cell,=experiment['cells']
    cell.update(cell_id=CELL,baseline_fragment=CELL,dataset_fragment=Path(inputs.DATASET).stem,
        run_id=RUN,attempt_id='never-flown',allowed_execution_stages=['check'],
        readiness={'state':'held','requirements':[],'reason':'Model-free agreement input only; no generation admission.'})
    experiment.pop('baseline_predecessor',None)
    experiment.update(experiment_id=RUN,split_id=read(repo/inputs.DATASET)['split_name'],
        purpose='Model-free R01C agreement preparation; no A0 gain or TFRL admission claim.',
        population={'count':625,'dataset_fragment':Path(inputs.DATASET).stem,
            'roster_receipt':inputs.ROSTER,'split_identity_hash':roster['split_identity_hash']},
        populations=[{'dataset_fragment':Path(inputs.DATASET).stem,'ids':ids}])
    experiment['stage3_measurement'].update(rows=625,read_roster=roster['roster_id'],claims_benefit=False)
    return baseline,experiment


def write(repo,source_repo,data,result_path,retrieval_complete,pool_audit,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    assert_execution_environment()
    for tree in (repo,source_repo):
        if subprocess.check_output(['git','-c','safe.directory='+str(tree),'status','--porcelain'],cwd=tree).strip():
            raise ValueError('clean source and destination required')
    result=read(result_path)
    if (result['schema']!='saea-real500-verified-result-v1' or result['rows']!=500
            or not result['official_finalized'] or not result['checkpoint_cost_reconciled']):
        raise ValueError('complete verified real500 source required')
    pin=result['selection_evidence']['manifest'];manifest=confined(data,pin['relpath'])
    if sha(manifest)!=pin['sha256']:raise ValueError('source manifest changed')
    bundle=verify_bundle(manifest)
    if (bundle.attempt_state!='FINALIZED' or len(bundle.outputs_rows)!=500
            or result['run_id']!=bundle.manifest['run_id']):
        raise ValueError('source bundle incomplete')
    if subprocess.check_output(['git','-c','safe.directory='+str(source_repo),'rev-parse','HEAD'],cwd=source_repo,text=True).strip()!=bundle.manifest['study_commit']:
        raise ValueError('source commit differs from completed method')
    fragments=bundle.manifest['config']['fragments']
    if compose(source_repo,**fragments).config_hash!=bundle.manifest['config']['config_hash']:
        raise ValueError('source configuration changed')
    prep=read(retrieval_complete);audit=read(pool_audit)
    if (prep['schema']!='saea-r01c-retrieval-inputs-v1' or prep['rows']!=625
            or not prep['values_unchanged'] or prep['encoder_run_id']!=inputs.RUN
            or audit['schema']!='saea-r01c-current-pool-audit-v1' or audit['utterance_overlap']!=0
            or audit['query_rows']!=625 or audit['covered_pool_facets']!=audit['facets']):
        raise ValueError('complete current R01C preparation required')
    for name,digest in prep['pins'].items():
        if sha(confined(data,name))!=digest:raise ValueError('retrieval input changed')
    base_rel=f"configs/baseline/{fragments['baseline']}.json"
    if (audit['pins']['source_baseline']!=sha(source_repo/base_rel)
            or audit['pins']['pool_receipt']!=sha(repo/R01C_RECEIPT_RELPATH)
            or audit['pins']['roster']!=sha(repo/inputs.ROSTER)):
        raise ValueError('pool audit source differs')
    pool_receipt=read(repo/R01C_RECEIPT_RELPATH)
    for key,rel in [('snapshot_file',pool_receipt['snapshot_artifact_relpath']),
                    ('release',pool_receipt['release_artifact_relpath'])]:
        if sha(confined(data,rel))!=audit['pins'][key]:raise ValueError('audited pool artifact changed')
    facet_pin=read(source_repo/base_rel)['knowledge_stores']['train-exemplars']['phase_supply']['solver']['facet_source']
    if sha(confined(data,facet_pin['relpath']))!=audit['pins']['facet_vectors']:
        raise ValueError('audited facet vectors changed')
    baseline,experiment=derive(repo,read(source_repo/base_rel),
        read(source_repo/f"configs/experiment/{fragments['experiment']}.json"),prep['pins'])
    outputs={f'configs/baseline/{CELL}.json':raw(baseline),f'configs/experiment/{CELL}.json':raw(experiment)}
    model_rel=f"configs/model/{fragments['model']}.json"
    source_model=read(source_repo/model_rel)
    for rel in (model_rel,'docs/'+source_model['runtime_receipt']):
        content=(source_repo/rel).read_bytes()
        if (repo/rel).exists() and (repo/rel).read_bytes()!=content:
            raise ValueError('destination model identity conflicts')
        if not (repo/rel).exists():outputs[rel]=content
    out=confined(data,str(out))
    if not out.is_relative_to((data/'work/codex').resolve()) or out.exists() or any((repo/p).exists() for p in outputs):
        raise ValueError('fresh output paths required')
    import hashlib
    record={'schema':'saea-r01c-current-comparator-v1','rows':625,'source_manifest_sha256':sha(manifest),
        'retrieval_completion_sha256':sha(retrieval_complete),'pool_audit_sha256':sha(pool_audit),
        'output_hashes':{p:hashlib.sha256(b).hexdigest() for p,b in outputs.items()},
        'model_contacts':0,'allowed_execution_stages':['check'],'claims_system_lock':False,'claims_tfrl_admitted':False}
    out.mkdir(parents=True);publish_new(out/'intent.json',raw(record))
    for rel,blob in outputs.items():
        path=repo/rel;publish_new(path,blob);path.chmod(0o644)
    publish_new(out/'complete.json',raw(record));return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','source-repo','data','result-path','retrieval-complete','pool-audit','out'):
        p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(write(**vars(p.parse_args()))))
