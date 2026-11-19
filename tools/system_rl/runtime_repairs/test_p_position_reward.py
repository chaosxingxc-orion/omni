"""Position reward through the shared scorer, Clock and GRPO reducer."""
import copy
import json
from pathlib import Path
import unittest

from fixtures import learning_config, bundle, receipt
from learning import evaluate
from scoring import score_component, score_episode, VERSION, validate_definition


class PositionRewardTests(unittest.TestCase):
    def config(self):
        config=learning_config('node','P')
        config['scorer']['version']=VERSION
        config['scorer']['metric_weights']['P']={'position_reward_v1':1.0}
        return config

    def test_position_and_coverage(self):
        for selected,expected in [(['a','b','c'],1),(['b','a','c'],.75),
                                  (['b','c','a'],.5),(['b','c','d'],0),([],0)]:
            part=score_component('P',{'retrieved_candidates':['a','b','c','d']},
                                 {'selected':selected},{'intent':'a'})
            self.assertEqual(part['metrics']['position_reward_v1'],expected)
            self.assertEqual(part['metrics']['coverage_at3'],int(expected>0))

    def test_business_others_is_not_a_failure_sentinel(self):
        for label in ('general_quirky','qa_factoid','others'):
            part=score_component('P',{'retrieved_candidates':['x',label]},
                                 {'selected':['x',label]},{'intent':label})
            self.assertEqual(part['metrics']['position_reward_v1'],.75)

    def test_bad_output_and_unknown_transport(self):
        for selected in (['a','a'],['outside'],['a','b','c','d']):
            item={'online':{'retrieved_candidates':['a','b','c','d']},
                  'output':{'selected':selected},'labels':{'intent':'a'},'call_ids':['P']}
            part=score_episode({'offline':{'P':item}},self.config()['scorer'])['P']
            self.assertEqual(part['value'],0)
            self.assertTrue(part['observed_model_failure'])
        with self.assertRaises(ValueError):
            score_component('P',item['online'],{'terminal':'transport_unknown'},item['labels'])
        self.assertEqual(score_component('P',item['online'],{},None)['mask'],0)

    def test_rank_signal_reaches_original_grpo(self):
        config=self.config();data,clock=bundle(config)
        rankings=[['a','b','c'],['b','a','c'],['b','c','a'],['b','c','d']]
        for episode,selected in zip(data['groups'][0]['episodes'],rankings):
            call=episode['calls'][0];call['output']['selected']=selected
            episode['offline']['P']['output']=copy.deepcopy(call['output'])
            receipt(call,episode['episode_id'])
        result=evaluate(data,clock)
        episodes=result['groups'][0]['episodes']
        self.assertEqual([e['reward'] for e in episodes],[1,.75,.5,0])
        adv=[e['advantage'] for e in episodes]
        self.assertTrue(all(a>b for a,b in zip(adv,adv[1:])))
        self.assertAlmostEqual(sum(adv),0)
        self.assertEqual(result['constant_reward_groups'],0)

    def test_old_definition_does_not_silently_change(self):
        config=self.config();config['scorer']['version']='component-definitions-v2'
        with self.assertRaises(ValueError):validate_definition(config['scorer'])
        config['scorer']['metric_weights']['P']={'coverage_at3':1.0}
        validate_definition(config['scorer'])

    def test_training_template_uses_one_p_objective(self):
        path=Path(__file__).resolve().parent.parent/'jev-system-rl/implementation/learning.template.json'
        config=json.loads(path.read_text())
        self.assertEqual(config['scorer']['version'],VERSION)
        self.assertEqual(config['scorer']['metric_weights']['P'],{'position_reward_v1':1.0})
        self.assertFalse(config['kernel']['ce']['enabled'])
        for phase in config['task_loss']['phases'].values():
            self.assertEqual(phase['P_ranking']['start_value'],0)
            self.assertEqual(phase['P_ranking']['end_value'],0)


if __name__=='__main__':unittest.main()
