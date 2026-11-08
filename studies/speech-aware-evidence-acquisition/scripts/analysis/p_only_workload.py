"""Source-bound P-only workload using the shared stage/prepare/register machinery."""
import copy
import json
import math
import re
from pathlib import Path

from intent_slot_workload import NodeWorkload
from tfrl_hardset_a0_registration import pinned
from tfrl_hardset_a0_workload import confined, sha
from gepa_pair_workload import read_asset

KIND = 'perception-only-native'
SCHEMA = 'saea-perception-only-workload-v1'
def verify_vector_selection(data, query, source_pin, ids):
    if not isinstance(source_pin, dict) or set(source_pin) != {'relpath', 'sha256'}:
        raise ValueError('explicit vector source pin required')
    original = confined(data, source_pin['relpath'])
    if sha(original) != source_pin['sha256']:
        raise ValueError('construction625 vector source differs')
    selected = set(ids)
    lines = {}
    with original.open('rb') as stream:
        for line in stream:
            item = json.loads(line)
            if item['id'] in selected:
                if item['id'] in lines:
                    raise ValueError('duplicate original vector')
                lines[item['id']] = line
    if len(lines) != len(ids) or confined(data, query['relpath']).read_bytes() != b''.join(lines[i] for i in ids):
        raise ValueError('query vectors are not exact construction625 selection')


def verify_ordered_supply(payload, ids, input_k, dynamic=False):
    from tfrl_odd40_cells import verify_output
    verify_output(payload, ids)
    if set(payload['facets']) != {'perception'}:
        raise ValueError('ordered facet rows required')
    for sample_id in ids:
        values = payload['values']['perception'][sample_id]
        facets = payload['facets']['perception'][sample_id]
        if (not isinstance(values, list) or (not 1 <= len(values) <= input_k if dynamic else len(values) != input_k)
                or any(not isinstance(v, str) or not v.startswith('intent-boundaries::intent-boundary:') for v in values)
                or len(set(values)) != len(values) or facets != values):
            raise ValueError('invalid ordered intent-boundary supply')


class PerceptionOnlyWorkload(NodeWorkload):
    LABEL = 'Fixed Fold A native Perception only'
    PROGRAM = 'Original audio Perception only; no Slot or Decision execution.'
    RESULT_SCHEMA = 'saea-perception-only-read-v1'

    def __init__(self, config):
        fields = {'schema', 'name', 'run_id', 'sample_ids', 'runtime_binding', 'execution',
                  'material', 'population', 'parent_baseline', 'parent_registration', 'axis'}
        if not fields <= set(config) <= fields | {'runtime_binding_adaptation', 'collection_decoding', 'learning_authorization', 'learning_source_vectors', 'learning_mode'} or config['schema'] != SCHEMA:
            raise ValueError('exact P-only configuration required')
        learning = 'learning_authorization' in config
        if learning:
            mode = config.get('learning_mode', 'tf-collection')
            if (config['population'].get('role') != 'construction' or len(config['sample_ids']) != 80
                    or len(set(config['sample_ids'])) != 80
                    or not isinstance(config.get('collection_decoding'), dict)
                    or set(config['collection_decoding']) != {'temperature', 'seed'}
                    or type(config['collection_decoding']['temperature']) not in (int, float)
                    or not math.isfinite(config['collection_decoding']['temperature'])
                    or (mode == 'tf-collection' and (config.get('axis') != {'kind': 'control'} or config['collection_decoding']['temperature'] <= 0))
                    or (mode == 'candidate-evaluation' and (not isinstance(config.get('axis'), dict) or config['axis'].get('kind') not in ('control', 'instruction', 'breadth', 'ranking-window') or config['collection_decoding']['temperature'] != 0))
                    or mode not in ('tf-collection', 'candidate-evaluation')
                    or not isinstance(config.get('learning_authorization'), dict)
                    or set(config['learning_authorization']) != {'source', 'source_roster'}
                    or not isinstance(config.get('learning_source_vectors'), dict)
                    or set(config['learning_source_vectors']) != {'perception', 'solver'}):
                raise ValueError('authorized construction80 learning mode and decoding required')
        elif ('learning_mode' in config or 'collection_decoding' in config or 'learning_source_vectors' in config or config['population'].get('dataset') != 'slurp-speech-native-evo-fold-a-1080' or len(config['sample_ids']) != 1080):
            raise ValueError('original Fold A 1080 required')
        axis = config['axis']
        if not isinstance(axis, dict) or axis.get('kind') not in ('control', 'instruction', 'breadth', 'knowledge-construction', 'ranking-window'):
            raise ValueError('explicit P-only axis required')
        if ((axis['kind'] == 'control' and set(axis) != {'kind'}) or
            (axis['kind'] == 'instruction' and set(axis) != {'kind', 'patch', 'template'}) or
            (axis['kind'] == 'knowledge-construction' and set(axis) != {'kind', 'publication', 'patch', 'audit'}) or
            (axis['kind'] == 'breadth' and (set(axis) != {'kind', 'value'} or type(axis['value']) is not int or not 1 <= axis['value'] <= 60)) or
            (axis['kind'] == 'ranking-window' and (not {'kind', 'input_k', 'output_k', 'max_tokens', 'template'} <= set(axis) <= {'kind', 'input_k', 'output_k', 'max_tokens', 'template', 'facet_source', 'transport', 'source_receipts', 'ordered_supply', 'dynamic'}
                or ('source_receipts' in axis and 'facet_source' not in axis)
                or ('ordered_supply' in axis and (axis.get('ordered_supply') not in axis.get('source_receipts', [])))
                or axis.get('transport', 'json-schema') not in {'json-schema', 'finite-choice-v1'}
                or ('dynamic' in axis and axis['dynamic'] is not True)
                or ((axis['input_k'], axis['output_k']) not in ((5, 1), (5, 3)) if axis.get('dynamic') else (axis['input_k'], axis['output_k']) not in ((3, 1), (3, 2), (5, 3)))
                or (axis.get('dynamic') and 'ordered_supply' not in axis)
                or type(axis['max_tokens']) is not int or not 1 <= axis['max_tokens'] <= 4096))):
            raise ValueError('invalid P-only axis')
        parent = {k: copy.deepcopy(v) for k, v in config.items() if k not in ('parent_baseline', 'parent_registration', 'axis', 'runtime_binding_adaptation', 'learning_authorization', 'learning_source_vectors', 'learning_mode')}
        parent.update(schema='saea-intent-slot-workload-v1', node_arm='enhanced-p-slot-decision')
        super().__init__(parent)
        self.config = copy.deepcopy(config)
        self.KIND = KIND
        self.ARMS = ['perception']
        self.DELIVERED_ARM = 'perception'
        self.CONTACTS = 6

    def prepare(self, repo, data, stage_record, out):
        axis = self.config['axis']
        if 'ordered_supply' not in axis:
            return super().prepare(repo, data, stage_record, out)
        from speech_aware_evidence_acquisition.contracts import assert_execution_environment
        from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
        from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
        from speech_aware_evidence_acquisition.core.preparation import prepare_artifact
        from speech_aware_evidence_acquisition.core.atomic_file import publish_new
        from tfrl_hardset_a0_workload import read, raw
        from tfrl_hardset_a0_registration import pin
        import tfrl_h1_observation as observation
        assert_execution_environment(); require_clean_study_repo(repo)
        observation.require_unspent(repo, data, self.RUN); self.fresh_output(data, out)
        staged = self.verify_stage(repo, data, stage_record)
        source = axis['ordered_supply']
        if source['root'] != 'data' or sha(confined(data, source['relpath'])) != source['sha256']:
            raise ValueError('ordered supply source differs')
        payload = read(confined(data, source['relpath']))
        verify_ordered_supply(payload, self.config['sample_ids'], axis['input_k'], axis.get('dynamic', False))
        runtime = build_flight_runtime(repo, data, model=staged['runtime_binding']['model_fragment'],
            dataset=self.DATASET, baseline=self.NAME, experiment=self.NAME, run_id=self.RUN, attempt_id='generation-v1',
            runtime_root=data/'work/scratch/runtime-root', lock_path=repo.parents[1]/'docs/datasets.lock.json')
        target = data/'prepared'/self.RUN/'phase-supply-v1.json'
        prepare_artifact(target, kind='phase-supply', inputs=runtime.phase_supply_artifact_identity()[-1], producer=lambda: payload)
        out.mkdir(parents=True)
        result = {'schema':'saea-gepa-pair-prepared-v1', 'stage_sha256':sha(stage_record), 'rows':self.ROWS,
            'run_id':self.RUN, 'supply':pin(data,str(target.relative_to(data))), 'new_model_calls':0}
        publish_new(out/'complete.json', raw(result)); return result

    def contact_budgets(self):
        return {'normal_generation_calls': self.ROWS, 'fallback_inclusive_cap': 6*self.ROWS,
                'fit_calls': 6*self.ROWS}

    def check_parent_binding(self, repo, original):
        current = self.config['runtime_binding']
        policy = self.config.get('runtime_binding_adaptation')
        if policy is None:
            if original != current:
                raise ValueError('P-only parent registration source differs')
            return
        if (not isinstance(policy, dict) or set(policy) != {'schema', 'comparison_scope', 'parent_model', 'current_model', 'parent_receipt', 'current_receipt'}
                or policy['schema'] != 'saea-p-only-runtime-binding-adaptation-v1'
                or policy['comparison_scope'] != 'same-model-and-request'):
            raise ValueError('exact P-only runtime binding adaptation required')
        models, receipts = [], []
        for label, binding in (('parent', original), ('current', current)):
            model_pin, receipt_pin = policy[label+'_model'], policy[label+'_receipt']
            if (not isinstance(binding, dict) or set(binding) != {'model_fragment', 'service_session', 'profile', 'runtime_receipt'}
                    or not isinstance(binding['model_fragment'], str)
                    or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', binding['model_fragment'])):
                raise ValueError('invalid P-only binding identity')
            if (not isinstance(binding['runtime_receipt'], str)
                    or not isinstance(model_pin, dict) or set(model_pin) != {'relpath', 'sha256'} or model_pin.get('relpath') != 'configs/model/'+binding['model_fragment']+'.json'
                    or not isinstance(receipt_pin, dict) or set(receipt_pin) != {'relpath', 'sha256'} or receipt_pin.get('relpath') != 'docs/'+binding['runtime_receipt']):
                raise ValueError('adaptation pins must identify bound model and receipt')
            model, receipt = pinned(repo, model_pin), pinned(repo, receipt_pin)
            if (not isinstance(model.get('model_lock_key'), str) or not model['model_lock_key']
                    or model.get('runtime_receipt') != binding['runtime_receipt'] or receipt.get('model_lock_key') != model['model_lock_key']):
                raise ValueError('adaptation model and receipt differ')
            files = receipt.get('model', {}).get('files')
            if not isinstance(files, list) or not files:
                raise ValueError('adaptation requires model file identities')
            identities = {}
            for item in files:
                if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'} or not isinstance(item['path'], str):
                    raise ValueError('invalid model file identity')
                name = Path(item['path']).name
                if (not name or name in identities or type(item['bytes']) is not int or item['bytes'] <= 0
                        or not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
                    raise ValueError('invalid model file identity')
                identities[name] = (item['bytes'], item['sha256'])
            if not any(name.endswith('.safetensors') for name in identities):
                raise ValueError('adaptation requires weight identities')
            receipts.append(identities)
            models.append({k: v for k, v in model.items() if k not in {'base_url', 'runtime_receipt', 'model_materialization_root', 'geometry_basis'}})
        if models[0] != models[1] or receipts[0] != receipts[1]:
            raise ValueError('P-only deployment model science or weights differ')

    def derive(self, repo, data, material_pin):
        material, checked = self.material_checked(repo, data, material_pin)
        binding = self.runtime_binding(repo)
        base = copy.deepcopy(pinned(repo, self.config['parent_baseline']))
        if (base.get('native_episode', {}).get('perception_only') is not True or
                base.get('thinking_policy', {}).get('perception', {}).get('enable_thinking') is not False):
            raise ValueError('pinned parent must be P-only with thinking OFF')
        parent_registration = pinned(repo, self.config['parent_registration'])
        original = parent_registration['job_config']
        learning = 'learning_authorization' in self.config
        if (parent_registration.get('workload_kind') != 'intent-slot-node-ablation'
                or any(original.get(k) != self.config[k] for k in (('execution',) if learning else ('sample_ids', 'population', 'material', 'execution')))):
            raise ValueError('P-only parent registration source differs')
        self.check_parent_binding(repo, original.get('runtime_binding'))
        from tfrl_hardset_a0_registration import scientific_hash
        parent_science = copy.deepcopy(base)
        parent_science['native_episode'].pop('perception_only')
        parent_science['native_contact_fit']['run_id'] = parent_registration['fit_run_id']
        parent_science['native_contact_fit']['attempt_id'] = parent_registration['fit_attempt_id']
        if scientific_hash(parent_science) != parent_registration['scientific_baseline_sha256']:
            raise ValueError('P-only parent differs from registered scientific baseline')
        if learning:
            authorization = pinned(repo, self.config['learning_authorization']['source'])
            ids = [g['query_id'] for g in authorization['document']['groups']]
            source_roster = pinned(repo, self.config['learning_authorization']['source_roster'])
            source_items = {item['sample_id']: item for item in source_roster['items']}
            if (ids != self.config['sample_ids'] or len(set(ids)) != 80
                    or source_roster.get('roster_role') != 'construction'
                    or source_roster.get('split_role') != 'discovery'
                    or len(source_items) != len(source_roster['items'])
                    or any(not i.startswith('slurp:train:') or i not in source_items for i in ids)
                    or set(ids) & set(original['sample_ids'])
                    or any(item != source_items[item['sample_id']] for item in checked['roster']['items'])):
                raise ValueError('authorized learning roster differs')
            old_source = pinned(repo, original['population']['source_baseline'])
            if old_source.get('policy') != 'phase-native-module-v1-c2':
                from derive_lock_only_baseline import derive as lock_only
                old_source = lock_only(old_source, repo=repo)
            new_source = copy.deepcopy(checked['source_baseline'])
            old_source = copy.deepcopy(old_source)
            for store, role in (('intent-boundaries', 'perception'), ('train-exemplars', 'solver')):
                verify_vector_selection(data, checked['source_baseline']['knowledge_stores'][store]['phase_supply'][role]['query_source'], self.config['learning_source_vectors'][role], ids)
                old_source['knowledge_stores'][store]['phase_supply'][role].pop('query_source')
                new_source['knowledge_stores'][store]['phase_supply'][role].pop('query_source')
            if old_source != new_source:
                raise ValueError('construction source science differs from parent')
            # Only the roster-bound query vectors change. All knowledge stores,
            # retrieval policy, templates and schema remain those of the parent.
            source = checked['source_baseline']
            for store, role in (('intent-boundaries', 'perception'), ('train-exemplars', 'solver')):
                base['knowledge_stores'][store]['phase_supply'][role]['query_source'] = copy.deepcopy(
                    source['knowledge_stores'][store]['phase_supply'][role]['query_source'])
        axis = self.config['axis']
        if axis['kind'] == 'instruction':
            from p_teacher_bridge import apply_parent_patch
            patch = pinned(repo, axis['patch'])
            template = axis['template']
            if not isinstance(template, dict) or set(template) != {'relpath', 'sha256'}:
                raise ValueError('pinned P instruction template required')
            path = confined(repo, template['relpath'])
            if not template['relpath'].startswith('configs/probes/') or sha(path) != template['sha256']:
                raise ValueError('P instruction template pin differs')
            base = apply_parent_patch(base, patch)
            if base['phase_material']['perception']['instruction_template'] != template['relpath'][len('configs/'):]:
                raise ValueError('patch and instruction template differ')
        elif axis['kind'] == 'knowledge-construction':
            from intent_snapshot_adapter import validate_published
            from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
            publication = copy.deepcopy(pinned(repo, axis['publication']))
            publication['data_root'] = str(data)
            for key in ('baseline', 'candidate', 'collected', 'context', 'parent_context', 'source'):
                item = publication[key]
                if set(item) != {'relpath', 'sha256'}:
                    raise ValueError('repository publication input pin required')
                pinned(repo, item)
                publication[key] = {'path': str(resolve_asset(repo, item['relpath'])), 'sha256': item['sha256']}
            base = validate_published(publication, base, axis['patch'], axis['audit'])
        elif axis['kind'] == 'breadth':
            supply = base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
            if (supply['k'], supply['fuse_coverage']['m'], supply['fuse_coverage']['top_l']) != (8, 8, 8):
                raise ValueError('parent retrieval breadth differs')
            from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot
            pin = base['knowledge_stores']['intent-boundaries']['snapshot']
            snapshot = KnowledgeSnapshot.from_frozen_json(json.loads(confined(data, pin['relpath']).read_bytes()))
            if snapshot.content_hash() != pin['sha256']:
                raise ValueError('intent inventory snapshot pin differs')
            dimensions = set(supply['dimensions'])
            inventory = sum(f.dimension in dimensions for f in snapshot.facets)
            if axis['value'] > inventory:
                raise ValueError('retrieval breadth exceeds pinned intent inventory')
            supply['k'] = supply['fuse_coverage']['m'] = supply['fuse_coverage']['top_l'] = axis['value']
        elif axis['kind'] == 'ranking-window':
            supply = base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
            if (supply['k'], supply['fuse_coverage']['m'], supply['fuse_coverage']['top_l']) != (8, 8, 8):
                raise ValueError('parent retrieval breadth differs')
            supply['k'] = supply['fuse_coverage']['m'] = supply['fuse_coverage']['top_l'] = axis['input_k']
            if 'facet_source' in axis:
                facet = axis['facet_source']
                if not isinstance(facet, dict) or set(facet) != set(supply['facet_source']) or facet.get('kind') != supply['facet_source'].get('kind'):
                    raise ValueError('ranking facet source must retain the pinned vector contract')
                if sha(confined(data, facet['relpath'])) != facet['sha256']:
                    raise ValueError('ranking facet source pin differs')
                from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot
                pin = base['knowledge_stores']['intent-boundaries']['snapshot']
                snapshot = KnowledgeSnapshot.from_frozen_json(json.loads(confined(data, pin['relpath']).read_bytes()))
                if snapshot.content_hash() != pin['sha256']:
                    raise ValueError('ranking intent snapshot pin differs')
                expected = {f.facet_id for f in snapshot.facets if f.dimension in supply['dimensions']}
                with confined(data, facet['relpath']).open() as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
                if len(rows) != len(expected) or {row['id'] for row in rows} != expected:
                    raise ValueError('ranking facet source must cover the exact intent inventory')
                supply['facet_source'] = copy.deepcopy(facet)
                if 'source_receipts' in axis:
                    receipts = axis['source_receipts']
                    if not isinstance(receipts, list) or not receipts:
                        raise ValueError('ranking source receipts must be nonempty')
                    existing = base.setdefault('preparation_source_receipts', [])
                    for receipt in receipts:
                        if (not isinstance(receipt, dict) or set(receipt) != {'root', 'relpath', 'sha256'}
                                or receipt['root'] != 'data'
                                or sha(confined(data, receipt['relpath'])) != receipt['sha256']
                                or any(r['root'] == receipt['root'] and r['relpath'] == receipt['relpath'] for r in existing)):
                            raise ValueError('ranking source receipt pin differs or duplicates')
                        existing.append(copy.deepcopy(receipt))
                if 'ordered_supply' in axis:
                    base['ordered_phase_supply'] = copy.deepcopy(axis['ordered_supply'])
            pipeline = base['native_episode']['intent_slot_pipeline']
            pipeline['max_candidates'] = axis['output_k']
            pipeline['ranking_window'] = {'mode':'ranked-candidate-indices', **{k:axis[k] for k in ('input_k','output_k','max_tokens','template')}}
            if axis.get('dynamic'):
                pipeline['ranking_window']['dynamic'] = True
            if 'transport' in axis:
                pipeline['ranking_window']['transport'] = axis['transport']
            if 'coverage_recovery' in pipeline:
                pipeline['coverage_recovery']['max_extra_contacts'] = min(pipeline['coverage_recovery']['max_extra_contacts'], axis['output_k'])
        base.pop('prepared_phase_supply', None)
        base['native_contact_fit'].update(run_id=self.RUN+'-FIT', attempt_id='fit-v1', contact_cap=6*self.ROWS)
        roster = checked['roster']
        roster['parent'] = self.config['population']['roster']
        roster['selection_rule'] = 'Reuse exact existing fold; no sampling or TEST tuning.'
        import tfrl_h1_observation as observation
        exp = read_asset(repo, observation.SOURCE_EXP)
        cell, = exp['cells']
        cell.update(cell_id=self.NAME, baseline_fragment=self.NAME, model_fragment=binding['model_fragment'],
                    dataset_fragment=self.DATASET, run_id=self.RUN, attempt_id='generation-v1',
                    allowed_execution_stages=['check', 'prepare-supply', 'worker', 'campaign'],
                    readiness={'state':'held', 'requirements':[], 'reason':'Registered P-only workload requires admission.'})
        exp.update(experiment_id=self.RUN, purpose=self.PROGRAM, service_session=binding['service_session'], split_id=checked['view']['split_name'])
        if learning:
            exp['request_decoding_overrides'] = copy.deepcopy(self.config['collection_decoding'])
            base['p_only_learning_scope'] = {
                'schema': 'saea-p-only-learning-scope-v1',
                'authorization': copy.deepcopy(self.config['learning_authorization']),
                'parent_baseline': copy.deepcopy(self.config['parent_baseline']),
                'parent_registration': copy.deepcopy(self.config['parent_registration']),
                'source_baseline': copy.deepcopy(self.config['population']['source_baseline']),
                'mode': self.config.get('learning_mode', 'tf-collection'),
            }
        exp['population'].update(count=self.ROWS, roster_receipt=self.ROSTER, split_identity_hash=roster['split_identity_hash'], dataset_fragment=self.DATASET)
        exp['populations'] = [{'dataset_fragment':self.DATASET, 'ids':self.config['sample_ids']}]
        exp['execution_plan'].update(planned_model_calls=6*self.ROWS, planned_speech_audio_seconds=math.ceil(6*math.fsum(r['audio_seconds'] for r in roster['items'])))
        exp['stage3_algorithm'] = {'id':'none', 'operators_used':[], 'note':'P-only intervention, no automatic acceptance.'}
        exp['stage3_measurement'].update(rows=self.ROWS, read_roster=roster['roster_id'], claims_benefit=False, expected_effect_pp=0, note=self.PROGRAM)
        return base, exp, roster, checked['view'], binding


def validate(repo, data, name, spec, exp, view):
    job = PerceptionOnlyWorkload(spec['job_config'])
    if spec['material'] != job.config['material']:
        raise ValueError('P-only material differs')
    return job.validate(repo, data, name, spec, exp, view)
