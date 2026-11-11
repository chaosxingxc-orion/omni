"""Configured knowledge provenance and source exclusion contracts."""
import copy
import unittest
from speech_aware_evidence_acquisition.rails.slot_stage_knowledge import build

class DefinitionTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{'slurp_id': 1, 'sentence': 'video games', 'tokens': [{'id': 0, 'surface': 'video'}], 'entities': [{'type': 'game_type', 'span': [0]}]}]
        self.bundle = {'bad_components': [], 'dev_components': [], 'source_components': {'1': 'source'}, 'snapshot': {'values': []}, 'original_sources': {}}
        self.policy = {'construction_split': 'train', 'kinds': {'game_type': 'enumerated_observed'}, 'closed_world_types': [], 'separation_groups': [], 'train_value_overlap_edges': True, 'max_types_per_fold': 8, 'semantic_definitions': {'game_type': {'definition': 'Kind of game.', 'source': 'reviewed-config', 'source_ids': [1]}}}

    def test_example_dependent_definition_quarantined_with_source(self):
        result = build(self.rows, self.bundle, self.policy, forbidden_components=['source'])
        self.assertEqual(result['cards']['game_type']['definition_status'], 'quarantined_semantic_definition')
        self.assertEqual(result['cards']['game_type']['examples'], [])
        self.assertEqual(result['report']['definition_gaps'], ['game_type'])

    def test_generic_definition_survives_without_reintroducing_example(self):
        self.policy['semantic_definitions']['game_type']['definition_depends_on_examples'] = False
        result = build(self.rows, self.bundle, self.policy, forbidden_components=['source'])
        card = result['cards']['game_type']
        self.assertEqual(card['definition'], 'Kind of game.')
        self.assertEqual(card['examples'], [])
        self.assertEqual(card['definition_components'], [])
        self.assertEqual(card['support_source_ids'], [1])
        self.assertEqual(result['report']['definition_gaps'], [])

    def test_missing_source_rejected(self):
        self.policy['semantic_definitions']['game_type']['source_ids'] = [999]
        with self.assertRaisesRegex(ValueError, 'missing TRAIN'):
            build(self.rows, self.bundle, self.policy)

if __name__ == '__main__':
    unittest.main()
