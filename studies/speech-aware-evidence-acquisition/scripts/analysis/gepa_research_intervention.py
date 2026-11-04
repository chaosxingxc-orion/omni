"""Pinned researcher-authored ablations of a verified external proposal.

These are research interventions, never newly generated teacher responses.
The actor contract, mutation syntax, tokenization and paired FIT stay intact.
"""
import json
import external_exploration as external
from gepa_external_mutation import verify_external_mutation
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha
from speech_aware_evidence_acquisition.rails.solver_prompt_candidate import validate_mutation


def verify_intervention(spec, *, contract, parent):
    if not isinstance(spec, dict) or set(spec) != {'schema', 'document'} or spec['schema'] != 'saea-gepa-research-intervention-v1':
        raise ValueError('exact research intervention pin required')
    path = external.checked(spec['document'])
    doc = external.read(path)
    fields = {'schema','source','parent_sha256','contract_sha256','mutation','review'}
    if set(doc) != fields or doc['schema'] != 'saea-gepa-research-intervention-document-v1':
        raise ValueError('exact intervention document required')
    source = verify_external_mutation(doc['source'], contract=contract, parent=parent)
    if doc['parent_sha256'] != parent['candidate_sha256'] or doc['contract_sha256'] != contract['contract_sha256']:
        raise ValueError('intervention actor parent or contract differs')
    review = doc['review']
    if (not isinstance(review,dict) or set(review) != {'author','kind','factor','rationale','source_mutation_sha256','mutation_sha256','decision'}
            or review['kind'] != 'researcher-authored-factor-ablation'
            or review['decision'] != 'eligible-for-feedback-only'
            or any(not isinstance(review[k],str) or not review[k].strip() for k in ('author','factor','rationale'))
            or review['source_mutation_sha256'] != sha(json.loads(source['mutation_text']))
            or review['mutation_sha256'] != sha(doc['mutation'])):
        raise ValueError('explicit source-bound researcher review required')
    mutation_text = canonical(doc['mutation']).decode()
    validate_mutation(mutation_text,contract=contract,expected_contract_sha256=contract['contract_sha256'],expected_parent_sha256=parent['candidate_sha256'])
    if mutation_text == source['mutation_text']:
        raise ValueError('intervention must change the source proposal')
    return {'mutation_text':mutation_text, 'response_sha256':spec['document']['sha256'],
        'contract_sha256':contract['contract_sha256'],'parent_sha256':parent['candidate_sha256'],
        'receipt_kind':'researcher-authored-factor-ablation-v1','source_external_response_sha256':source['response_sha256'],
        'intervention_sha256':spec['document']['sha256'],'review':review,
        'new_model_calls':0,'candidate_accepted':False,'native_fit_completed':False}
