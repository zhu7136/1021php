import tempfile,unittest,json
from pathlib import Path
from types import SimpleNamespace
import torch
from hud04_action_probe import SameStateActionProbe,tensor_stats

class ProbeTests(unittest.TestCase):
    def fixture(self,root):
        policy=SimpleNamespace(routing_column=449)
        contract={'widths':{'actor_obs':{'a':90,'b':6}},'offsets':{'actor_obs':{'a':0,'b':90}}}
        probe=SameStateActionProbe(policy,lambda x:torch.ones(1,29),root,contract)
        privileged=torch.zeros(1,450);privileged[0,449]=2
        args={'step':1,'frame':0,'actor':torch.arange(96.).reshape(1,96),'teacher':privileged,
              'depth':torch.ones(1,58,87),'student_actions':torch.zeros(1,29),'expected_route':2}
        return probe,args

    def test_stats_reports_flat_zero(self):
        stats=tensor_stats(torch.zeros(1,58,87))
        self.assertEqual(stats['zero_fraction'],1.)
        self.assertEqual(stats['std'],0.)

    def test_known_error_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder)
            before={k:v.clone() for k,v in args.items() if torch.is_tensor(v)}
            probe.observe(**args);result=probe.finish()
            self.assertEqual(result['aggregate_action_rmse'],1.)
            self.assertEqual(result['per_joint_rmse'],[1.]*29)
            for k,v in before.items():self.assertTrue(torch.equal(args[k],v))
            self.assertTrue((Path(folder)/'selected-state-inputs.npz').is_file())
            row=json.loads((Path(folder)/'same-state-actions.jsonl').read_text())
            self.assertEqual(row['actor_terms']['b']['min'],90.)

    def test_wrong_route_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);args['teacher'][0,449]=3
            with self.assertRaises(ValueError):probe.observe(**args)

    def test_nonfinite_depth_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);args['depth'][0,0,0]=float('nan')
            with self.assertRaises(ValueError):probe.observe(**args)

    def test_nonfinite_teacher_action_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);probe.teacher_forward=lambda x:torch.full((1,29),float('nan'))
            with self.assertRaises(ValueError):probe.observe(**args)

    def test_depth_changes_and_two_step_rmse(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);probe.observe(**args)
            args.update(step=2,frame=1,depth=torch.zeros(1,58,87),student_actions=torch.ones(1,29))
            probe.observe(**args);result=probe.finish()
            self.assertAlmostEqual(result['aggregate_action_rmse'],2**-.5)
            self.assertEqual(result['depth_changed_observation_pairs'],1)

    def test_no_samples_not_success_or_zero_error(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);result=probe.finish()
            self.assertIsNone(result['aggregate_action_rmse'])
            self.assertFalse(result['teacher_actions_applied'])

    def test_repeated_step_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);probe.observe(**args)
            with self.assertRaises(ValueError):probe.observe(**args)

    def test_masked_route_skips_teacher_without_changing_observations(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);args['teacher'][0,449]=-1
            before=args['teacher'].clone()
            def forbidden(_):raise AssertionError('Masked teacher must not be evaluated')
            probe.teacher_forward=forbidden
            probe.observe(**args);result=probe.finish()
            self.assertTrue(torch.equal(args['teacher'],before))
            self.assertEqual(result['steps'],1)
            self.assertEqual(result['expert_invalid_steps'],1)
            self.assertEqual(result['compared_steps'],0)
            self.assertIsNone(result['aggregate_action_rmse'])
            record=json.loads((Path(folder)/'same-state-actions.jsonl').read_text())
            self.assertEqual(record['route'],-1)
            self.assertFalse(record['comparison_valid'])
            self.assertIsNone(record['teacher_actions'])

    def test_masked_step_excluded_from_denominator_and_next_step_continues(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);probe.observe(**args)
            args.update(step=2);args['teacher'][0,449]=-1;probe.observe(**args)
            args.update(step=3);args['teacher'][0,449]=2;probe.observe(**args)
            result=probe.finish()
            self.assertEqual(result['steps'],3)
            self.assertEqual(result['compared_steps'],2)
            self.assertEqual(result['action_metric_denominator_steps'],2)
            self.assertEqual(result['aggregate_action_rmse'],1.)

    def test_unknown_negative_fractional_routes_rejected(self):
        for route in [-2,1.5,0,4]:
            with self.subTest(route=route),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                probe,args=self.fixture(folder);args['teacher'][0,449]=route
                with self.assertRaises(ValueError):probe.observe(**args)

    def test_nonfinite_inputs_still_rejected_when_teacher_masked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            probe,args=self.fixture(folder);args['teacher'][0,449]=-1
            args['student_actions'][0,0]=float('inf')
            with self.assertRaises(ValueError):probe.observe(**args)

if __name__=='__main__':unittest.main()