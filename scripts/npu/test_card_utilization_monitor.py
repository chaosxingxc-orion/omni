import unittest
from copy import deepcopy
from card_utilization_monitor import summarize, assess_windows


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'time': f'2026-11-06T00:00:{second:02d}Z',
                      'cards': {str(c): {'ai': value} for c in range(8)}}
                     for second, value in [(0, 100), (10, 100), (40, 60), (50, 100)]]
        self.config = {'schema': 'saea-utilization-windows-v1',
                       'threshold_percent': 80, 'max_sample_gap_seconds': 30,
                       'windows': [{'id': name, 'cards': cards,
                                    'first_dispatch': '2026-11-06T00:00:00Z',
                                    'last_response': '2026-11-06T00:00:50Z',
                                    'boundary_evidence_sha256': 'a'*64}
                                   for name, cards in [('A', list(range(4))), ('B', list(range(4, 8)))]]}

    def test_time_weighting_includes_idle_and_drain(self):
        result = assess_windows(self.rows, self.config)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['windows'][0]['cards']['0']['time_weighted_mean_ai_percent'], 84)
        self.rows[2]['cards']['0']['ai'] = 0
        result = assess_windows(self.rows, self.config)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['windows'][0]['cards']['0']['time_weighted_mean_ai_percent'], 60)
        self.assertEqual(result['windows'][1]['cards']['4']['status'], 'PASS')

    def test_exact_threshold_does_not_pass(self):
        for row in self.rows:
            for value in row['cards'].values(): value['ai'] = 80
        self.assertEqual(assess_windows(self.rows, self.config)['status'], 'FAIL')

    def test_missing_card_gap_and_unbracketed_window_are_unknown(self):
        for problem in ('card', 'gap', 'start', 'end', 'nan'):
            rows, config = deepcopy(self.rows), deepcopy(self.config)
            if problem == 'card': del rows[1]['cards']['7']
            if problem == 'gap': config['max_sample_gap_seconds'] = 20
            if problem == 'start': rows = rows[1:]
            if problem == 'end': rows = rows[:-1]
            if problem == 'nan': rows[1]['cards']['0']['ai'] = float('nan')
            with self.subTest(problem=problem):
                self.assertEqual(assess_windows(rows, config)['status'], 'INSUFFICIENT_EVIDENCE')

    def test_separate_windows_exclude_only_outside_observations(self):
        self.config['windows'][1]['first_dispatch'] = '2026-11-06T00:00:10Z'
        self.config['windows'][1]['last_response'] = '2026-11-06T00:00:40Z'
        result = assess_windows(self.rows, self.config)
        self.assertEqual(result['windows'][1]['cards']['4']['time_weighted_mean_ai_percent'], 80)
        self.assertEqual(result['status'], 'FAIL')

    def test_invalid_identity_or_order_refused(self):
        for problem in ('duplicate', 'overlap', 'seven_cards', 'naive_time', 'no_evidence'):
            rows, config = deepcopy(self.rows), deepcopy(self.config)
            if problem == 'duplicate': rows.insert(1, deepcopy(rows[0]))
            if problem == 'overlap': config['windows'][1]['cards'][0] = 0
            if problem == 'seven_cards': config['windows'][1]['cards'].pop()
            if problem == 'naive_time': rows[0]['time'] = '2026-11-06T00:00:00'
            if problem == 'no_evidence': config['windows'][0]['boundary_evidence_sha256'] = ''
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                assess_windows(rows, config)


class SummaryTests(unittest.TestCase):
    def test_missing_is_not_idle_and_all_cards_must_meet_threshold(self):
        rows=[{'cards':{0:{'ai':100},1:{'ai':0}}},
              {'cards':{0:{'ai':90},1:{'ai':95}}},
              {'cards':{0:{'ai':50}}}]
        r=summarize(rows,[0,1])
        self.assertEqual(r['complete_samples'],2)
        self.assertEqual(r['all_cards_above_threshold_fraction'],0.5)
        self.assertEqual(r['cards']['1']['missing_samples'],1)
        self.assertEqual(r['cards']['1']['zero_ai_fraction'],0.5)
        self.assertEqual(r['cards']['1']['mean_ai_percent'],47.5)

    def test_no_observation_never_claims_utilization(self):
        r=summarize([{'cards':{}}],list(range(8)))
        self.assertIsNone(r['all_cards_above_threshold_fraction'])
        self.assertTrue(all(c['mean_ai_percent'] is None for c in r['cards'].values()))


if __name__=='__main__':unittest.main()
