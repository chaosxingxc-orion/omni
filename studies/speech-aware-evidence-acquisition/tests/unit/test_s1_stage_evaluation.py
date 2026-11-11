import asyncio
import copy
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
CODE = STUDY / 'scripts/analysis/s1_stage_evaluation'
spec = importlib.util.spec_from_file_location('public_s1_score', CODE / 'score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
# Fixed public partition fixture, with no data or reference transcripts.
FOLDS = [{'fold_id': 'fold-00', 'types': ['business_name', 'color_type', 'email_folder', 'event_name', 'food_type', 'game_type']}, {'fold_id': 'fold-01', 'types': ['alarm_type', 'app_name', 'ingredient', 'list_name', 'person', 'transport_type']}, {'fold_id': 'fold-02', 'types': ['artist_name', 'currency_name', 'date', 'definition_word', 'player_setting', 'playlist_name']}, {'fold_id': 'fold-03', 'types': ['change_amount', 'coffee_type', 'music_genre', 'order_type', 'place_name', 'podcast_name']}, {'fold_id': 'fold-04', 'types': ['device_type', 'house_place', 'news_topic', 'song_name', 'sport_type', 'time_zone']}, {'fold_id': 'fold-05', 'types': ['general_frequency', 'joke_type', 'meal_type', 'radio_name', 'transport_agency']}, {'fold_id': 'fold-06', 'types': ['audiobook_name', 'movie_type', 'relation', 'time', 'transport_descriptor']}, {'fold_id': 'fold-07', 'types': ['audiobook_author', 'cooking_type', 'game_name', 'music_descriptor', 'timeofday']}, {'fold_id': 'fold-08', 'types': ['business_type', 'movie_name', 'personal_info', 'podcast_descriptor', 'weather_descriptor']}, {'fold_id': 'fold-09', 'types': ['drink_type', 'email_address', 'media_type', 'music_album', 'transport_name']}]
TYPES = [t for f in FOLDS for t in f['types']]


def prediction(sid, selected=(), omit=(), override=None):
    branches = []
    for f in FOLDS:
        if f['fold_id'] in omit:
            continue
        picked = sorted(set(selected) & set(f['types']))
        b = {'stage': 'S1', 'sample_id': sid, 'branch': f['fold_id'],
             'status': 'valid_nonempty' if picked else 'valid_empty',
             'selected_types': picked, 'finish_reason': 'stop',
             'request_id': sid + ':' + f['fold_id'],
             'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}
        b.update((override or {}).get(f['fold_id'], {}))
        branches.append(b)
    union = sorted({t for b in branches for t in b['selected_types']})
    return {'sample_id': sid, 'complete': not omit, 's1': branches, 'selected_types': union}


def run(gold, preds):
    return score.score([{'sample_id': sid, 'gold_types': ts} for sid, ts in gold.items()], preds, FOLDS)[0]


class ScoreTests(unittest.TestCase):
    def test_global_55_types_and_10_folds(self):
        self.assertEqual(len(TYPES), 55)
        self.assertEqual(len(FOLDS), 10)
        self.assertEqual(len(set(TYPES)), 55)

    def test_real_counts_not_fixed_expected_size(self):
        s = run({'one': ['time', 'date'], 'empty': []}, [prediction('one', ['time', 'date']), prediction('empty')])
        self.assertEqual(s['metrics']['type_set_exact_accuracy'], 1)
        self.assertEqual(s['metrics']['positive_complete_type_coverage'], 1)
        self.assertEqual(s['metrics']['empty_gold_correct_rate'], 1)

    def test_failure_and_absent_empty_do_not_get_credit(self):
        p = prediction('failed', override={'fold-00': {'status': 'protocol_error'}})
        s = run({'failed': [], 'absent': []}, [p])
        self.assertEqual(s['metrics']['type_set_exact_accuracy'], 0)
        self.assertEqual(s['metrics']['empty_gold_correct_rate'], 0)
        self.assertEqual(s['metrics']['all_sample_complete_type_coverage'], 0)
        self.assertIsNone(s['metrics']['positive_complete_type_coverage'])
        self.assertEqual(s['fold_status_counts']['absent'], 10)

    def test_partial_fold_cannot_earn_complete_coverage_even_if_all_gold_observed(self):
        p = prediction('p', ['time'], omit=['fold-00'])
        s = run({'p': ['time']}, [p])
        self.assertEqual(s['metrics']['positive_observed_type_coverage'], 1)
        self.assertEqual(s['metrics']['positive_complete_type_coverage'], 0)
        self.assertEqual(s['metrics']['type_set_exact_accuracy'], 0)

    def test_duplicate_types_deduplicated(self):
        p = prediction('p', ['time'])
        b = next(b for b in p['s1'] if 'time' in b['selected_types'])
        b['selected_types'] *= 2
        s = run({'p': ['time', 'time']}, [p])
        self.assertEqual(s['metrics']['type_set_exact_accuracy'], 1)
        self.assertEqual(s['metrics']['micro_observed']['true_positive'], 1)

    def test_duplicate_fold_not_complete(self):
        p = prediction('p')
        p['s1'].append(copy.deepcopy(p['s1'][0]))
        s = run({'p': []}, [p])
        self.assertEqual(s['counts']['complete_samples'], 0)
        self.assertEqual(s['fold_status_counts']['duplicate_branch'], 1)

    def test_unknown_and_truncation_keep_denominator(self):
        p = prediction('p', override={'fold-00': {'status': 'unknown'}, 'fold-01': {'status': 'incomplete', 'finish_reason': 'length'}})
        s = run({'p': [], 'q': ['time']}, [p])
        self.assertEqual(s['denominator'], 2)
        self.assertEqual(s['fold_status_counts']['unknown'], 1)
        self.assertEqual(s['fold_status_counts']['incomplete'], 1)
        self.assertEqual(s['metrics']['micro_observed']['missing'], 1)

    def test_union_mismatch_and_outside_fold_rejected(self):
        p = prediction('p')
        p['s1'][0]['selected_types'] = ['time']
        p['selected_types'] = ['time']
        s = run({'p': ['time']}, [p])
        self.assertEqual(s['counts']['aggregate_protocol_error_samples'], 1)
        self.assertEqual(s['metrics']['positive_complete_type_coverage'], 0)

    def test_count_equal_can_still_be_wrong_set(self):
        s = run({'p': ['date']}, [prediction('p', ['time'])])
        self.assertEqual(s['metrics']['type_count_match_accuracy'], 1)
        self.assertEqual(s['metrics']['type_set_exact_accuracy'], 0)
        self.assertEqual(s['metrics']['mean_reference_unmatched_candidates'], 1)
        self.assertEqual(s['metrics']['mean_missing_types'], 1)

    def test_positive_main_coverage_not_inflated_by_empty(self):
        s = run({'positive': ['time'], 'empty': []}, [prediction('positive'), prediction('empty')])
        self.assertEqual(s['metrics']['positive_complete_type_coverage'], 0)
        self.assertEqual(s['metrics']['all_sample_complete_type_coverage'], .5)

    def test_per_type_macro_support_excludes_unseen(self):
        s = run({'p': ['time']}, [prediction('p', ['date'])])
        self.assertEqual(len(s['per_type']), 55)
        self.assertEqual(s['metrics']['macro_supported_types_observed']['supported_types'], 1)
        self.assertEqual(s['metrics']['macro_supported_types_observed']['f1'], 0)
        self.assertEqual(s['per_type']['date']['reference_unmatched'], 1)

    def test_paid_invalid_usage_preserved(self):
        p = prediction('p', override={'fold-00': {'status': 'protocol_error'}})
        s = run({'p': []}, [p])
        self.assertEqual(s['cost']['usage']['total_tokens'], 120)
        self.assertEqual(s['cost']['unique_observed_requests'], 10)
        self.assertIsNone(s['cost']['money'])

    def test_stage_timing_survives_transport_timing_and_missing_is_unknown(self):
        p = prediction('p')
        p['s1'][0]['stage_timing'] = {'call_and_rails_seconds': 3.5}
        p['s1'][0]['timing'] = {'new_http_seconds': 2.0}
        s = run({'p': []}, [p])
        self.assertEqual(s['cost']['summed_call_and_rails_seconds'], 3.5)
        self.assertEqual(s['cost']['summed_observed_new_http_seconds'], 2.0)
        s = run({'p': []}, [prediction('p')])
        self.assertIsNone(s['cost']['summed_call_and_rails_seconds'])

    def test_truncation_reported_independently(self):
        p = prediction('p', override={'fold-00': {'status': 'incomplete', 'finish_reason': 'length'}})
        s = run({'p': []}, [p])
        self.assertEqual(s['truncated_fold_results'], 1)
        self.assertEqual(s['counts']['complete_samples'], 0)

    def test_unknown_duplicate_predictions_fail_fast(self):
        with self.assertRaises(ValueError):
            run({'p': []}, [prediction('p'), prediction('p')])
        with self.assertRaises(ValueError):
            run({'p': []}, [prediction('q')])

    def test_null_and_equal_type_fillers_preserve_types(self):
        p = prediction('p', ['time', 'date'])
        gold = [{'sample_id': 'p', 'gold_types': ['time', 'date'],
                 'gold_entities': [{'type': 'time', 'filler': None}, {'type': 'date', 'filler': 'date'}]}]
        s, _ = score.score(gold, [p], FOLDS)
        self.assertEqual(s['metrics']['type_set_exact_accuracy'], 1)

    def test_real_slotstages_aggregate_is_consumed(self):
        # Load production rails without importing unrelated model/framework init.
        src = STUDY / 'src/speech_aware_evidence_acquisition/rails'
        pkg = types.ModuleType('s1_under_test')
        pkg.__path__ = [str(src)]
        sys.modules['s1_under_test'] = pkg
        spec = importlib.util.spec_from_file_location('s1_under_test.slot_stages', src / 'slot_stages.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        knowledge = {'cards': {t: {'type': t, 'definition': t, 'examples': []} for t in TYPES},
                     'folds': FOLDS, 'source_components': {}, 'knowledge_id': 'fixture'}
        cfg = json.loads((src.parents[2] / 'configs/slots/s-stages-v1/runtime.json').read_bytes())
        rails = mod.StageRails(knowledge, cfg)
        async def caller(packet, after):
            text = json.dumps({'slot_types': ['time'] if 'time' in packet['allowed_types'] else []})
            return after(packet, {'text': text, 'finish_reason': 'stop', 'usage': {'total_tokens': 7}})
        stages = mod.SlotStages(rails, caller)
        query = mod.Query('slurp:devel:000001:audio-unit', {'type': 'input_audio', 'input_audio': {'data': 'fixture', 'format': 'flac'}})
        aggregate = asyncio.run(stages.run_s1(query))
        s = run({query.sample_id: ['time']}, [aggregate])
        self.assertEqual(s['status'], 'COMPLETE')
        self.assertEqual(s['metrics']['type_set_exact_accuracy'], 1)
        self.assertEqual(s['cost']['usage']['total_tokens'], 70)


if __name__ == '__main__':
    unittest.main(verbosity=2)
