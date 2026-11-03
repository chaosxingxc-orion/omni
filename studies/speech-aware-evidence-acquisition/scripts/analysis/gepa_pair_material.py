"""Compile paired templates only from finalized native mutation provenance."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import gepa_seed_material as seed
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails import solver_prompt_candidate as prompt
from speech_aware_evidence_acquisition.rails.gepa_prompt_material import compile_template_pair
from speech_aware_evidence_acquisition.rails.tokenizer_runtime import load_text_token_counter
from tfrl_odd40_cells import read,raw,sha,confined
SCHEMA='saea-gepa-pair-material-v1'


def derive(repo,data,*,seed_material,mutation_manifest):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.rails.gepa_mutation_runtime import verify_mutation_bundle
    from speech_aware_evidence_acquisition.rails.gepa_mutation_source import seed_contract
    checked=seed_contract({**seed_material,'root':'repository' if seed_material['root']=='repo' else 'data'},repo=repo,data=data)
    seed_doc=checked['document']
    path,_=seed.read_pin(repo,data,mutation_manifest)
    native=verify_mutation_bundle(verify_bundle(path),repo=repo,data=data)
    contract,parent=checked['contract'],checked['parent']
    if native['contract_sha256']!=contract['contract_sha256'] or native['parent_sha256']!=parent['candidate_sha256']:
        raise ValueError('mutation belongs to another frozen parent or contract')
    count,tokenizer=load_text_token_counter(Path(checked['runtime_receipt']['model']['model_dir']))
    try:
        if tokenizer!=checked['tokenizer_identity']:raise ValueError('paired tokenizer changed')
        child=prompt.derive_candidate(native['mutation_text'],parent=parent,
            expected_parent_sha256=parent['candidate_sha256'],contract=contract,
            expected_contract_sha256=contract['contract_sha256'],count_tokens=count)
        common=dict(contract=contract,expected_contract_sha256=contract['contract_sha256'],count_tokens=count,
            tokenizer_identity=tokenizer,expected_tokenizer_sha256=prompt.sha(tokenizer))
        original=confined(repo/'configs',checked['source_baseline']['phase_material']['solver']['instruction_template']).read_text()
        compiled=compile_template_pair(repo,checked['source_baseline'],parent_arguments=dict(common,candidate=parent),
            child_arguments=dict(common,candidate=child,parent=parent,expected_parent_sha256=parent['candidate_sha256'],mutation_text=native['mutation_text']),
            fit_estimate=seed_doc['fit_estimate'],max_supplied_items=seed_doc['max_supplied_items'],delivering_template_tokens=count(original))
    finally:
        if hasattr(count,'close'):count.close()
    return {'compiled':compiled,'contract':contract,'parent':parent,'child':child,'source_baseline':checked['source_baseline'],
        'model':checked['model'],'partition':checked['partition'],'source_pins':seed_doc['source_pins'],'mutation':native,'seed_material':seed_doc}


def build(repo,data,spec,out):
    if set(spec)!={'schema','seed_material','mutation_manifest'} or spec['schema']!='saea-gepa-pair-build-v1':
        raise ValueError('exact paired build specification required')
    if out.exists() or not out.resolve().is_relative_to((repo/'configs').resolve()):raise ValueError('fresh config output required')
    result=derive(repo,data,seed_material=spec['seed_material'],mutation_manifest=spec['mutation_manifest']);out.mkdir(parents=True)
    docs={k:result[k] for k in ('contract','parent','child')}
    docs.update(compiled_baseline=result['compiled']['baseline'],declaration=result['compiled']['declaration'])
    for name,doc in docs.items():publish_new(out/(name+'.json'),raw(doc))
    manifest={'schema':SCHEMA,'seed_material':spec['seed_material'],'mutation_manifest':spec['mutation_manifest'],
        'documents':{k:seed.pin(repo,out/(k+'.json')) for k in docs},
        'parent_sha256':result['parent']['candidate_sha256'],'child_sha256':result['child']['candidate_sha256'],
        'mutation_native_response_sha256':result['mutation']['response_sha256'],
        'normal_calls_per_row':4,'fallback_cap_per_row':8,'new_model_calls':0,'candidate_accepted':False}
    publish_new(out/'complete.json',raw(manifest));return manifest


def verify(repo,data,material):
    if material.get('schema')!=SCHEMA:raise ValueError('paired material required')
    expected=derive(repo,data,seed_material=material['seed_material'],mutation_manifest=material['mutation_manifest'])
    docs={k:seed.read_pin(repo,data,v)[1] for k,v in material['documents'].items()}
    want={k:expected[k] for k in ('contract','parent','child')}
    want.update(compiled_baseline=expected['compiled']['baseline'],declaration=expected['compiled']['declaration'])
    if docs!=want or material['parent_sha256']!=expected['parent']['candidate_sha256'] or material['child_sha256']!=expected['child']['candidate_sha256']:
        raise ValueError('pair differs from authenticated mutation replay')
    if (material['mutation_native_response_sha256']!=expected['mutation']['response_sha256']
            or material['normal_calls_per_row']!=4 or material['fallback_cap_per_row']!=8
            or material['new_model_calls']!=0 or material['candidate_accepted'] is not False):
        raise ValueError('paired material source budget or claim changed')
    return {**expected,**docs}


if __name__=='__main__':
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','spec','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert_execution_environment();print(json.dumps(build(a.repo,a.data,read(a.spec),a.out),sort_keys=True))
