"""Static annotations of already verified TRAIN candidates; never release rules.

Call only after the native source verifier. A structural annotation is an
explicit hypothesis about an item, not a verified translation of its prose.
"""
import copy

from .slot_error_feedback import _frame
from .slot_rule_admission import POOL_SCHEMA, review_slot_rule, require
from .tfgrpo_learning import sha

SCHEMA = 'saea-slot-rule-annotations-v1'
RESULT_SCHEMA = 'saea-slot-rule-review-v1'


def review_annotations(annotations, *, candidate, candidate_pin, document, source_manifest):
    require(isinstance(annotations, dict) and set(annotations) == {
        'schema', 'candidate', 'document_sha256', 'items'}
        and annotations['schema'] == SCHEMA, 'exact slot annotations required')
    require(annotations['candidate'] == candidate_pin
            and annotations['document_sha256'] == sha(document),
            'annotations must bind exact candidate and verified source document')
    items = annotations['items']
    require(isinstance(items, dict) and bool(items)
            and set(items) <= set(candidate['items']), 'annotations must name candidate items')
    rows = []
    groups = document['groups']
    require(isinstance(groups, list) and bool(groups), 'nonempty verified TRAIN groups required')
    for group in groups:
        key = group['query_id']
        require(isinstance(key, str) and key.startswith('slurp:train:'),
                'only original TRAIN groups may supply slot evidence')
        reference = group['evidence'].get('reference')
        rows.append({'evidence_id': key, 'source_sha256': sha(group),
            'reference_complete': _frame(reference) is not None,
            'reference': copy.deepcopy(reference)})
    pool = {'schema': POOL_SCHEMA, 'split': 'train', 'rows': rows}
    results = {}
    for key, annotation in items.items():
        require(isinstance(annotation, dict) and set(annotation) == {'content_sha256', 'rule'}
                and annotation['content_sha256'] == sha(candidate['items'][key]),
                'annotation item content changed')
        require(isinstance(annotation['rule'], dict), 'explicit structured rule required')
        results[key] = {'content_sha256': annotation['content_sha256'],
            'rule_sha256': sha(annotation['rule']),
            'semantic_translation_verified': False,
            'static_result': review_slot_rule(annotation['rule'], pool, expected_pool_sha256=sha(pool))}
    return {'schema': RESULT_SCHEMA, 'candidate': copy.deepcopy(candidate_pin),
        'source_manifest': copy.deepcopy(source_manifest), 'document_sha256': sha(document),
        'annotations_sha256': sha(annotations), 'evidence_pool_sha256': sha(pool),
        'evidence_pool': pool, 'items': results,
        'unannotated_items': sorted(set(candidate['items']) - set(items)),
        'native_source_verified_by_exporter': True,
        'semantic_support_verified': False, 'candidate_deployment_accepted': False,
        'independent_review_required': True, 'new_model_calls': 0}
