import unittest
from card_utilization_monitor import summarize


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
