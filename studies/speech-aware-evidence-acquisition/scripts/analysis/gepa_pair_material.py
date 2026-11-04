"""Compile paired templates from verified native or external mutation provenance."""
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

def source_digest_field(spec):
    return ('mutation_intervention_sha256' if 'intervention' in spec else
            'mutation_external_response_sha256' if 'external_mutation' in spec else 'mutation_native_response_sha256')


def derive(repo,data,*,seed_material,mutation_manifest=None,historical_repository=None,mutation_repository=None,external_mutation=None, intervention=None):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.rails.gepa_mutation_runtime import verify_mutation_bundle
    from speech_aware_evidence_acquisition.rails.gepa_mutation_source import seed_contract,source_repository
    historical=source_repository({} if historical_repository is None else {'historical_repository':historical_repository},repo=repo,data=data)
    mutation_repo=source_repository({} if mutation_repository is None else {'historical_repository':mutation_repository},repo=repo,data=data)
    checked=seed_contract({**seed_material,'root':'repository' if seed_material['root']=='repo' else 'data'},repo=historical,data=data)
    seed_doc=checked['document']
    # The mutation's registered model/receipt and fragments belong to its sealed
    # source tree. Never recompose them against a different live serving lane.
    contract,parent=checked['contract'],checked['parent']
    if sum(x is not None for x in (mutation_manifest, external_mutation, intervention)) != 1:
        raise ValueError('exactly one native or external mutation source required')
    if intervention is not None:
        from gepa_research_intervention import verify_intervention
        native=verify_intervention(intervention,contract=contract,parent=parent)
    elif external_mutation is not None:
        from gepa_external_mutation import verify_external_mutation
        native=verify_external_mutation(external_mutation,contract=contract,parent=parent)
    else:
        path,_=seed.read_pin(mutation_repo,data,mutation_manifest)
        native=verify_mutation_bundle(verify_bundle(path),repo=mutation_repo,data=data)
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
        template=checked['source_baseline']['phase_material']['solver']['instruction_template']
        original=confined(historical/'configs',template).read_text()
        if confined(repo/'configs',template).read_bytes()!=confined(historical/'configs',template).read_bytes():
            raise ValueError('live Solver template differs from historical source')
        compiled=compile_template_pair(repo,checked['source_baseline'],parent_arguments=dict(common,candidate=parent),
            child_arguments=dict(common,candidate=child,parent=parent,expected_parent_sha256=parent['candidate_sha256'],mutation_text=native['mutation_text']),
            fit_estimate=seed_doc['fit_estimate'],max_supplied_items=seed_doc['max_supplied_items'],delivering_template_tokens=count(original))
    finally:
        if hasattr(count,'close'):count.close()
    return {'compiled':compiled,'contract':contract,'parent':parent,'child':child,'source_baseline':checked['source_baseline'],
        'model':checked['model'],'partition':checked['partition'],'source_pins':seed_doc['source_pins'],'mutation':native,'seed_material':seed_doc}


def build(repo,data,spec,out):
    if not {'schema','seed_material'}<=set(spec)<={'schema','seed_material','mutation_manifest','external_mutation','intervention','historical_repository','mutation_repository'} or spec['schema']!='saea-gepa-pair-build-v1':
        raise ValueError('exact paired build specification required')
    if sum(k in spec for k in ('mutation_manifest','external_mutation','intervention')) != 1:
        raise ValueError('exactly one mutation source required')
    if out.exists() or not out.resolve().is_relative_to((repo/'configs').resolve()):raise ValueError('fresh config output required')
    result=derive(repo,data,seed_material=spec['seed_material'],mutation_manifest=spec.get('mutation_manifest'),external_mutation=spec.get('external_mutation'),intervention=spec.get('intervention'),historical_repository=spec.get('historical_repository'),mutation_repository=spec.get('mutation_repository'));out.mkdir(parents=True)
    docs={k:result[k] for k in ('contract','parent','child')}
    docs.update(compiled_baseline=result['compiled']['baseline'],declaration=result['compiled']['declaration'])
    for name,doc in docs.items():publish_new(out/(name+'.json'),raw(doc))
    manifest={'schema':SCHEMA,'seed_material':spec['seed_material'],**({k:spec[k] for k in ('mutation_manifest','external_mutation','intervention') if k in spec}),
        **({'historical_repository':spec['historical_repository']} if 'historical_repository' in spec else {}),
        **({'mutation_repository':spec['mutation_repository']} if 'mutation_repository' in spec else {}),
        'documents':{k:seed.pin(repo,out/(k+'.json')) for k in docs},
        'parent_sha256':result['parent']['candidate_sha256'],'child_sha256':result['child']['candidate_sha256'],
        source_digest_field(spec):result['mutation']['response_sha256'],
        'normal_calls_per_row':4,'fallback_cap_per_row':8,'new_model_calls':0,'candidate_accepted':False}
    publish_new(out/'complete.json',raw(manifest));return manifest


def verify(repo,data,material):
    if material.get('schema')!=SCHEMA:raise ValueError('paired material required')
    expected=derive(repo,data,seed_material=material['seed_material'],mutation_manifest=material.get('mutation_manifest'),external_mutation=material.get('external_mutation'),intervention=material.get('intervention'),historical_repository=material.get('historical_repository'),mutation_repository=material.get('mutation_repository'))
    docs={k:seed.read_pin(repo,data,v)[1] for k,v in material['documents'].items()}
    want={k:expected[k] for k in ('contract','parent','child')}
    want.update(compiled_baseline=expected['compiled']['baseline'],declaration=expected['compiled']['declaration'])
    if docs!=want or material['parent_sha256']!=expected['parent']['candidate_sha256'] or material['child_sha256']!=expected['child']['candidate_sha256']:
        raise ValueError('pair differs from authenticated mutation replay')
    if (material[source_digest_field(material)]!=expected['mutation']['response_sha256']
            or material['normal_calls_per_row']!=4 or material['fallback_cap_per_row']!=8
            or material['new_model_calls']!=0 or material['candidate_accepted'] is not False):
        raise ValueError('paired material source budget or claim changed')
    return {**expected,**docs}


if __name__=='__main__':
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','spec','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert_execution_environment();print(json.dumps(build(a.repo,a.data,read(a.spec),a.out),sort_keys=True))
