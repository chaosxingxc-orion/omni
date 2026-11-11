"""Public template API and actual-wire change boundary."""
import copy
import unittest
from test_slot_stages import CONFIG, knowledge, query
from speech_aware_evidence_acquisition.rails.slot_stages import StageRails, unified_templates, digest
from speech_aware_evidence_acquisition.rails.configured_s1 import protocol_wire_change, scientific_changes


class ProtocolConfigTest(unittest.TestCase):
    def test_explicit_default_wire_and_template_hash_bound(self):
        config = copy.deepcopy(CONFIG)
        config.update(s1_candidate_presentation='unified_candidates_v1', s1_none_label='OTHER_SLOT_TYPE')
        parent = StageRails(knowledge(), config)
        templates = unified_templates(config)
        child_config = {**config, 's1_unified_templates': {**templates, 'sha256': digest(templates)}}
        child = StageRails(knowledge(), child_config)
        self.assertEqual(parent.before_model_call('S1', query(), 'f1')['wire'], child.before_model_call('S1', query(), 'f1')['wire'])
        templates['user_instruction'] = templates['user_instruction'].replace('one or more', 'zero or more')
        child_config['s1_unified_templates'] = {**templates, 'sha256': digest(templates)}
        child = StageRails(knowledge(), child_config)
        before, after = parent.before_model_call('S1', query(), 'f1'), child.before_model_call('S1', query(), 'f1')
        self.assertNotEqual(before['request_id'], after['request_id'])
        changed = scientific_changes({'allowed_changes': ['s1_unified_templates']}, config, child_config)
        protocol_wire_change(before['wire'], after['wire'], config, child_config, changed)
        altered = copy.deepcopy(after['wire']); altered['max_tokens'] += 1
        with self.assertRaises(ValueError): protocol_wire_change(before['wire'], altered, config, child_config, changed)
        child_config['s1_unified_templates']['sha256'] = '0'*64
        with self.assertRaises(ValueError): StageRails(knowledge(), child_config)
