import unittest
from types import SimpleNamespace as NS
import torch
from hud04_tracking_cache_sync import extract_cache_block,install_cache_sync

SOURCE='''
def step(self):
    self.time_steps += 1
    use_root = self.time_steps * 0
    self.body_quat_relative_w = self.body_quat_w.clone()
    self.body_pos_relative_w = self.body_pos_w + use_root[:, None, None]
    self.sampler_updates += 1
'''

class CacheSyncTests(unittest.TestCase):
    def setUp(self):
        self.command=NS(time_steps=torch.tensor([0]),body_pos_w=torch.tensor([[[105.,3005.,.1]]]),
            body_quat_w=torch.tensor([[[0.,0.,0.,1.]]]),body_pos_relative_w=torch.zeros(1,1,3),sampler_updates=0)
        self.calls=0
        def original():self.calls+=1;return self.command.body_pos_relative_w.abs().max()>1000
        self.env=NS(num_envs=1,is_evaluating=True,command_manager=NS(get_state=lambda _:self.command),_check_termination=original)
        self.refresh=extract_cache_block(SOURCE,{'torch':torch})
    def test_extract_excludes_clock_and_sampler(self):
        self.refresh(self.command);self.assertEqual(self.command.time_steps.item(),0);self.assertEqual(self.command.sampler_updates,0)
    def test_targets_updated(self):
        self.refresh(self.command);self.assertTrue(torch.equal(self.command.body_pos_relative_w,self.command.body_pos_w))
    def test_native_check_runs_once_after_refresh(self):
        state=install_cache_sync(self.env,self.refresh);self.assertTrue(self.env._check_termination())
        self.assertEqual(self.calls,1);self.assertEqual(state['checks'],1)
    def test_multiple_native_checks_refresh_current_frame(self):
        install_cache_sync(self.env,self.refresh);self.env._check_termination()
        self.command.time_steps.fill_(7);self.command.body_pos_w.fill_(42);self.env._check_termination()
        self.assertEqual(self.command.time_steps.item(),7);self.assertEqual(self.calls,2)
        self.assertTrue((self.command.body_pos_relative_w==42).all())
    def test_training_forbidden(self):
        self.env.is_evaluating=False
        with self.assertRaises(ValueError):install_cache_sync(self.env,self.refresh)
    def test_training_switch_forbidden(self):
        install_cache_sync(self.env,self.refresh);self.env.is_evaluating=False
        with self.assertRaises(PermissionError):self.env._check_termination()
        self.assertEqual(self.calls,0)
    def test_multiple_envs_forbidden(self):
        self.env.num_envs=2
        with self.assertRaises(ValueError):install_cache_sync(self.env,self.refresh)
    def test_duplicate_install_forbidden(self):
        install_cache_sync(self.env,self.refresh)
        with self.assertRaises(RuntimeError):install_cache_sync(self.env,self.refresh)
    def test_unrelated_mutation_blocked(self):
        source=SOURCE.replace('use_root = self.time_steps * 0','use_root = self.time_steps * 0\n    self.other_state = 1')
        with self.assertRaises(ValueError):extract_cache_block(source,{'torch':torch})
    def test_clock_mutation_guard(self):
        def invalid(c):c.time_steps+=1
        install_cache_sync(self.env,invalid)
        with self.assertRaises(RuntimeError):self.env._check_termination()
        self.assertEqual(self.calls,0)

if __name__=='__main__':unittest.main()