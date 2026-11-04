"""Explicit Perception GEPA seed/release from pinned source files and wire custody.

Configuration and immutable evidence are inputs. Never edits live trees or calls
models. Native registration, FIT and root approval remain separate steps.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from speech_aware_evidence_acquisition.rails import perception_prompt_candidate as prompt
from speech_aware_evidence_acquisition.rails.external_exploration import checked, pin, read
from speech_aware_evidence_acquisition.rails.tokenizer_runtime import load_text_token_counter
from speech_aware_evidence_acquisition.rails.gepa_prompt_material import compile_template_pair
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
import gepa_seed_material as seed
from tfrl_odd40_cells import confined, raw

SEED_SCHEMA='saea-gepa-perception-seed-v1'
SCHEMA='saea-gepa-perception-pair-material-v1'


def derive_seed(repo, spec):
    fields={'schema','source_baseline','model','partition','tokenizer_identity','tokenizer_path',
            'feedback_query_ids','excluded_query_ids','fit_estimate','max_supplied_items','source_pins'}
    if set(spec)!=fields or spec['schema']!=SEED_SCHEMA:
        raise ValueError('exact perception seed specification required')
    source={k:read(checked(spec[k])) for k in ('source_baseline','model','partition','tokenizer_identity')}
    base=source['source_baseline']
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    cm.verify_declared_module(base['native_episode']['consumption_module'],base,repo_root=repo)
    if base['native_episode']['consumption_module']['columns']:
        raise ValueError('unvaried baseline required')
    for key,value in spec['source_pins'].items():checked(value)
    partition=source['partition']
    if not set(spec['feedback_query_ids']) <= set(partition['feedback_ids']):
        raise ValueError('feedback exposure must remain in original feedback partition')
    if not set(partition['selection_ids']) <= set(spec['excluded_query_ids']):
        raise ValueError('original selection remains excluded')
    count,tokenizer=load_text_token_counter(Path(spec['tokenizer_path']))
    try:
        if tokenizer!=source['tokenizer_identity']:raise ValueError('actor tokenizer changed')
        frozen=seed.frozen_fields(repo,base,source['model'],tokenizer)
        frozen.pop('perception_sha256')
        solver=base['phase_material']['solver']
        frozen['solver_sha256']=prompt.sha({'material':solver,
            'template_sha256':seed.sha(confined(repo/'configs',solver['instruction_template'])),
            'role':base['native_episode']['roles']['solver'],'thinking':base['thinking_policy']['solver']})
        # Both node output interfaces remain immutable; P editing cannot change its schema.
        frozen['output_schema_sha256']=prompt.sha({node:{'material':base['phase_material'][node],
            'template_sha256':seed.sha(confined(repo/'configs',base['phase_material'][node]['instruction_template']))}
            for node in ('perception','solver')})
        path=confined(repo/'configs',base['phase_material']['perception']['instruction_template'])
        original=path.read_text();first,rest=original.split('\n\n',1)
        contract=prompt.build_contract(base_template=prompt.BLOCK_START+first+prompt.BLOCK_END+'\n\n'+rest,
            frozen=frozen,feedback_query_ids=spec['feedback_query_ids'],excluded_query_ids=spec['excluded_query_ids'])
        parent=prompt.seed_candidate(contract=contract,expected_contract_sha256=contract['contract_sha256'],count_tokens=count)
        return dict(contract=contract,parent=parent,source_baseline=base,model=source['model'],
            partition=partition,tokenizer_identity=tokenizer,original=original)
    finally:
        if hasattr(count,'close'):count.close()


def build_seed(repo,spec,out):
    result=derive_seed(repo,spec)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('contract','parent','tokenizer_identity'):
        publish_new(out/(name+'.json'),raw(result[name]))
    manifest={'schema':SEED_SCHEMA,'specification':spec,
        'documents':{name:pin(out/(name+'.json')) for name in ('contract','parent','tokenizer_identity')},
        'new_model_calls':0,'candidate_accepted':False}
    publish_new(out/'complete.json',raw(manifest));return manifest


def derive(repo,data,spec):
    if set(spec) not in ({'schema','seed','external_mutation'},
                        {'schema','seed','external_mutation','repeat_scope'}) or spec['schema']!='saea-gepa-perception-pair-build-v1':
        raise ValueError('exact perception pair build configuration required')
    manifest=read(checked(spec['seed']))
    if manifest['schema']!=SEED_SCHEMA:raise ValueError('explicit perception seed required')
    result=derive_seed(repo,manifest['specification'])
    for name,value in manifest['documents'].items():
        if read(checked(value))!=result[name]:raise ValueError('seed replay differs')
    from gepa_external_mutation import verify_external_mutation
    mutation=verify_external_mutation(spec['external_mutation'],contract=result['contract'],parent=result['parent'])
    count,tokenizer=load_text_token_counter(Path(manifest['specification']['tokenizer_path']))
    try:
        contract,parent=result['contract'],result['parent']
        child=prompt.derive_candidate(mutation['mutation_text'],parent=parent,
            expected_parent_sha256=parent['candidate_sha256'],contract=contract,
            expected_contract_sha256=contract['contract_sha256'],count_tokens=count)
        common=dict(contract=contract,expected_contract_sha256=contract['contract_sha256'],count_tokens=count,
            tokenizer_identity=tokenizer,expected_tokenizer_sha256=prompt.sha(tokenizer))
        compiled=compile_template_pair(repo,result['source_baseline'],parent_arguments=dict(common,candidate=parent),
            child_arguments=dict(common,candidate=child,parent=parent,expected_parent_sha256=parent['candidate_sha256'],mutation_text=mutation['mutation_text']),
            fit_estimate=manifest['specification']['fit_estimate'],max_supplied_items=manifest['specification']['max_supplied_items'],
            delivering_template_tokens=count(result['original']),
            perception_repeat_scope=spec.get('repeat_scope','shared-lock-perception'))
    finally:
        if hasattr(count,'close'):count.close()
    return {**result,'child':child,'compiled_baseline':compiled['baseline'],'declaration':compiled['declaration'],
        'mutation':mutation,'source_pins':{'partition':manifest['specification']['partition']}}


def build(repo,data,spec,out):
    result=derive(repo,data,spec)
    out.mkdir(parents=True,exist_ok=False)
    documents={}
    for name in ('contract','parent','child','compiled_baseline','declaration'):
        publish_new(out/(name+'.json'),raw(result[name]));documents[name]=pin(out/(name+'.json'))
    manifest={'schema':SCHEMA,'build':spec,'documents':documents,'normal_calls_per_row':8,
        'fallback_cap_per_row':16,'new_model_calls':0,'candidate_accepted':False,
        'external_request_sha256':result['mutation']['request_sha256']}
    publish_new(out/'complete.json',raw(manifest));return manifest


def verify(repo,data,manifest):
    if manifest['schema']!=SCHEMA:raise ValueError('explicit perception material required')
    result=derive(repo,data,manifest['build'])
    if set(manifest['documents'])!={'contract','parent','child','compiled_baseline','declaration'}:
        raise ValueError('exact perception documents required')
    for name,value in manifest['documents'].items():
        if read(checked(value))!=result[name]:raise ValueError('perception material replay differs')
    if (manifest['normal_calls_per_row']!=8 or manifest['fallback_cap_per_row']!=16
            or manifest['new_model_calls']!=0 or manifest['candidate_accepted'] is not False
            or manifest['external_request_sha256']!=result['mutation']['request_sha256']):
        raise ValueError('perception cost or source claim differs')
    return result


if __name__=='__main__':
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['seed','pair'])
    for key in ('repo','data','spec','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();assert_execution_environment()
    result=build_seed(a.repo,read(a.spec),a.out) if a.action=='seed' else build(a.repo,a.data,read(a.spec),a.out)
    print(json.dumps({'schema':result['schema'],'new_model_calls':0}))
