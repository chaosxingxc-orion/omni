"""The registered construction80 P-only source and actor boundary."""
import hashlib
import json
import re
from dataclasses import dataclass

from ..core.experiment_assets import resolve_asset


def _read(repo, pin):
    if not isinstance(pin, dict) or set(pin) != {'relpath', 'sha256'}:
        raise ValueError('P learning requires exact source pins')
    raw = resolve_asset(repo, pin['relpath']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('P learning source pin changed')
    return json.loads(raw)


def _utterance(sample_id):
    parts = sample_id.split(':')
    return ':'.join(parts[:3])


def _normal(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


@dataclass(frozen=True)
class PerceptionLearningScope:
    ids: frozenset
    reference_text: dict
    binding_sha256: str
    snapshot: object
    forbidden_content: tuple

    def assert_snapshot(self, snapshot, sample_id):
        if sample_id not in self.ids or snapshot is not self.snapshot:
            raise ValueError('P learning query or snapshot differs from registered scope')

    def assert_visible(self, query_state):
        ids = (*query_state.retrieved_value_ids, *query_state.visible_facet_ids)
        if any(not item.startswith('intent-boundaries::') for item in ids):
            raise ValueError('P learning visible source is not intent-boundaries')

    def assert_request(self, prepared, sample_id):
        if sample_id not in self.ids:
            raise ValueError('P learning request outside authorized roster')
        role = prepared.role_request
        if role.role_id != 'perception' or role.visible_tool_names or role.tool_schemas:
            raise ValueError('P learning request grants an unauthorized role or tool')
        body = prepared.rendered_body()
        if body.get('tools') or body.get('tool_schemas'):
            raise ValueError('P learning request renders unauthorized tools')
        text = _normal(json.dumps(body.get('messages', []), ensure_ascii=False))
        if any(len(reference) >= 32 and len(reference.split()) >= 5 and reference in text
               for reference in self.reference_text.values()):
            raise ValueError('P learning request contains teacher reference')
        encoded = json.dumps(body.get('messages', []), ensure_ascii=False).casefold()
        if 'reference_feedback_private' in encoded or 'teacher_reference' in encoded:
            raise ValueError('P learning request contains teacher reference field')
        if 'train exemplars' in text:
            raise ValueError('P learning request contains exemplar source')
        for value in self.forbidden_content:
            if len(value) >= 16 and value in text:
                raise ValueError('P learning request contains unauthorized source content')


def load_scope(repo, data, values, plan, store_binding, native_config, experiment):
    declaration = values.get('p_only_learning_scope')
    if declaration is None:
        return None
    if (not isinstance(declaration, dict) or set(declaration) != {'schema', 'authorization', 'parent_baseline', 'parent_registration', 'source_baseline', 'mode'}
            or declaration['schema'] != 'saea-p-only-learning-scope-v1'
            or declaration['mode'] not in ('tf-collection', 'candidate-evaluation')
            or not native_config.perception_only or native_config.role('perception').tool_names
            or native_config.constant_tool_names or native_config.relisten is not None
            or native_config.knowledge_index_json is not None
            or native_config.perception_branch_group is not None
            or values.get('phase_tools') or store_binding is None):
        raise ValueError('invalid P-only learning capability')
    auth = declaration['authorization']
    if not isinstance(auth, dict) or set(auth) != {'source', 'source_roster'}:
        raise ValueError('original P learning authorization required')
    source = _read(repo, auth['source'])
    roster = _read(repo, auth['source_roster'])
    parent = _read(repo, declaration['parent_registration'])
    _read(repo, declaration['parent_baseline'])
    _read(repo, declaration['source_baseline'])
    # Re-run the existing complete registration validator, including derivation,
    # parent science, roster, model, prepared supply and runtime pins.
    import sys
    scripts = str(repo / 'scripts' / 'analysis')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from registered_workload import validate_workload
    candidate = json.loads(resolve_asset(repo, f'docs/readiness/{experiment}/candidate.json').read_bytes())
    if candidate.get('workload_kind') != 'perception-only-native' or candidate.get('run_id') != plan.run_id:
        raise ValueError('P learning requires its registered workload')
    validate_workload(repo, data, experiment, candidate)
    groups = source['document']['groups']
    ids = [g['query_id'] for g in groups]
    if (len(ids) != 80 or len(set(ids)) != 80 or any(not x.startswith('slurp:train:') for x in ids)
            or roster.get('roster_role') != 'construction' or roster.get('split_role') != 'discovery'
            or not set(ids) <= {x['sample_id'] for x in roster['items']}
            or parent.get('workload_kind') != 'intent-slot-node-ablation'
            or candidate['job_config']['sample_ids'] != ids
            or candidate['job_config']['learning_authorization'] != auth
            or set(ids) & set(parent['job_config']['sample_ids'])):
        raise ValueError('P learning authorization or parent population differs')
    store = store_binding.store('intent-boundaries')
    if (store.store_role != 'intent-boundaries' or store.snapshot.source_partition_role != 'train'
            or {_utterance(x) for x in ids} & {_utterance(x) for x in store.snapshot.source_sample_ids}):
        raise ValueError('P learning intent source overlaps query')
    if (any(name != 'intent-boundaries' and 'perception' in item.phase_supply
            for name, item in store_binding.stores.items())
            or any('perception' in item.candidate_supply or 'perception' in item.context_supply
                   for item in store_binding.stores.values())):
        raise ValueError('P learning perception route exposes unauthorized store')
    references = {}
    for group in groups:
        query_id = group['query_id']
        feedback = group['evidence']['reference_feedback_private']
        if group['evidence']['query']['sample_id'] != query_id or not feedback.get('reference_sentence'):
            raise ValueError('P learning reference identity differs')
        references[query_id] = _normal(feedback['reference_sentence'])
    utterances = {_utterance(x) for x in ids}
    other = tuple(_normal(value.content) for name in store_binding.store_ids if name != 'intent-boundaries'
                  for value in store_binding.store(name).snapshot.values
                  if utterances & {_utterance(x) for x in value.source_sample_ids})
    return PerceptionLearningScope(frozenset(ids), references, store_binding.binding_sha256,
                                   store_binding.composite_snapshot, other)
