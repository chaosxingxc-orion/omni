"""Source-bound P-only workload using the shared stage/prepare/register machinery."""
import copy
import math
from pathlib import Path

from intent_slot_workload import NodeWorkload
from tfrl_hardset_a0_registration import pinned
from tfrl_hardset_a0_workload import confined, sha
from gepa_pair_workload import read_asset

KIND = 'perception-only-native'
SCHEMA = 'saea-perception-only-workload-v1'


class PerceptionOnlyWorkload(NodeWorkload):
    LABEL = 'Fixed Fold A native Perception only'
    PROGRAM = 'Original audio Perception only; no Slot or Decision execution.'
    RESULT_SCHEMA = 'saea-perception-only-read-v1'

    def __init__(self, config):
        fields = {'schema', 'name', 'run_id', 'sample_ids', 'runtime_binding', 'execution',
                  'material', 'population', 'parent_baseline', 'parent_registration', 'axis'}
        if set(config) != fields or config['schema'] != SCHEMA:
            raise ValueError('exact P-only configuration required')
        if config['population'].get('dataset') != 'slurp-speech-native-evo-fold-a-1080' or len(config['sample_ids']) != 1080:
            raise ValueError('original Fold A 1080 required')
        axis = config['axis']
        if not isinstance(axis, dict) or axis.get('kind') not in ('control', 'instruction', 'breadth'):
            raise ValueError('explicit P-only axis required')
        if ((axis['kind'] == 'control' and set(axis) != {'kind'}) or
            (axis['kind'] == 'instruction' and set(axis) != {'kind', 'patch', 'template'}) or
            (axis['kind'] == 'breadth' and (set(axis) != {'kind', 'value'} or type(axis['value']) is not int or axis['value'] not in (12, 16)))):
            raise ValueError('invalid P-only axis')
        parent = {k: copy.deepcopy(v) for k, v in config.items() if k not in ('parent_baseline', 'parent_registration', 'axis')}
        parent.update(schema='saea-intent-slot-workload-v1', node_arm='enhanced-p-slot-decision')
        super().__init__(parent)
        self.config = copy.deepcopy(config)
        self.KIND = KIND
        self.ARMS = ['perception']
        self.DELIVERED_ARM = 'perception'
        self.CONTACTS = 6

    def contact_budgets(self):
        return {'normal_generation_calls': self.ROWS, 'fallback_inclusive_cap': 6*self.ROWS,
                'fit_calls': 6*self.ROWS}

    def derive(self, repo, data, material_pin):
        material, checked = self.material_checked(repo, data, material_pin)
        binding = self.runtime_binding(repo)
        base = copy.deepcopy(pinned(repo, self.config['parent_baseline']))
        if (base.get('native_episode', {}).get('perception_only') is not True or
                base.get('thinking_policy', {}).get('perception', {}).get('enable_thinking') is not False):
            raise ValueError('pinned parent must be P-only with thinking OFF')
        parent_registration = pinned(repo, self.config['parent_registration'])
        original = parent_registration['job_config']
        if (parent_registration.get('workload_kind') != 'intent-slot-node-ablation'
                or any(original.get(k) != self.config[k] for k in ('sample_ids', 'population', 'material', 'runtime_binding', 'execution'))):
            raise ValueError('P-only parent registration source differs')
        from tfrl_hardset_a0_registration import scientific_hash
        parent_science = copy.deepcopy(base)
        parent_science['native_episode'].pop('perception_only')
        parent_science['native_contact_fit']['run_id'] = parent_registration['fit_run_id']
        parent_science['native_contact_fit']['attempt_id'] = parent_registration['fit_attempt_id']
        if scientific_hash(parent_science) != parent_registration['scientific_baseline_sha256']:
            raise ValueError('P-only parent differs from registered scientific baseline')
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
        elif axis['kind'] == 'breadth':
            supply = base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
            if (supply['k'], supply['fuse_coverage']['m'], supply['fuse_coverage']['top_l']) != (8, 8, 8):
                raise ValueError('parent retrieval breadth differs')
            supply['k'] = supply['fuse_coverage']['m'] = supply['fuse_coverage']['top_l'] = axis['value']
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
