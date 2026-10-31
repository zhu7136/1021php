"""Real CPU tensor/Adam round-trip; actual pinned native logging path guard.

No simulator import, GPU use, or training workload. Toy tensors test persistence.
"""
import ast
import copy
import json
import logging
import os
from pathlib import Path
import tempfile
from types import MethodType,SimpleNamespace
import unittest
import torch
from three_adapter.checkpoint_export import finalize_native_checkpoint,NAME

ROOT=Path(__file__).resolve().parent
LOGGING_SOURCE=Path('/workspace/low-climb-discovery/cpu-payload/bundle/runtime/holosoma/src/holosoma/holosoma/agents/modules/logging_utils.py')
tree=ast.parse(LOGGING_SOURCE.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='save_checkpoint_artifact' for m in n.body))
method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint_artifact'))
# Resolve no upstream module imports: only this exact pinned method is compiled.
method.returns=None
for a in method.args.args:a.annotation=None
scope={'os':os,'torch':torch,'logger':logging.getLogger('native-checkpoint-cpu-test')}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'pinned_native_logging_guard','exec'),scope)

def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def require_equal(a,b):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or not torch.equal(a.detach().cpu(),b.detach().cpu()):raise RuntimeError('Tensor mismatch')
    elif isinstance(a,dict):
        if set(a)!=set(b):raise RuntimeError('Key mismatch')
        for k,v in a.items():require_equal(v,b[k])
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise RuntimeError('Sequence mismatch')
        for x,y in zip(a,b):require_equal(x,y)
    elif a!=b:raise RuntimeError('Scalar mismatch')

class FakeAlgo:
    def __init__(self,output):
        self.log_dir=str(output/'HuD04_WBT/timestamped-validation')
        Path(self.log_dir).mkdir(parents=True)
        self.logging_helper=SimpleNamespace(log_dir=self.log_dir,save_to_wandb=lambda p:None)
        self.logging_helper.save_checkpoint_artifact=MethodType(scope['save_checkpoint_artifact'],self.logging_helper)
        self.policy=torch.nn.Module()
        for key in ('student','critic','depth_backbone'):setattr(self.policy,key,torch.nn.Linear(2,2))
        self.policy.std=torch.nn.Parameter(torch.ones(2))
        self.policy.loaded_teachers=[True]*3
        self.optimizer=torch.optim.Adam([{'params':list(getattr(self.policy,k).parameters())} for k in ('student','critic','depth_backbone')]+[{'params':[self.policy.std]}],lr=3e-4)
        # One toy CPU operation initializes real Adam slots for persistence tests.
        sum(p.square().sum() for p in self.policy.parameters()).backward();self.optimizer.step();self.optimizer.zero_grad()
        self.current_learning_iteration=4;self.learning_rate=3e-4;self.ppo_coef=0.0004
        self.fault=None
    def save(self,path):
        if self.fault=='save':raise ValueError('Injected native save failure')
        saved={'model':self.policy.state_dict(),'optimizer':self.optimizer.state_dict(),
            'iter':self.current_learning_iteration,'lr':self.learning_rate,'ppo':self.ppo_coef,'ready':self.policy.loaded_teachers}
        self.logging_helper.save_checkpoint_artifact(saved,path)
    def load(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        self.policy.load_state_dict(saved['model']);self.optimizer.load_state_dict(saved['optimizer'])
        self.current_learning_iteration=saved['iter'];self.learning_rate=saved['lr'];self.ppo_coef=saved['ppo'];self.policy.loaded_teachers=saved['ready']
        if self.fault=='model':
            with torch.no_grad():self.policy.std.add_(1)
        elif self.fault=='optimizer':self.optimizer.param_groups[0]['lr']*=2
        elif self.fault=='scalar':self.ppo_coef*=2
        elif self.fault=='ready':self.policy.loaded_teachers[0]=False
        elif self.fault=='bytes':
            with Path(path).open('ab') as f:f.write(b'changed')

class CheckpointExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='checkpoint-cpu-',dir=ROOT);self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name);self.algo=FakeAlgo(self.output)
    def run_export(self):return finalize_native_checkpoint(self.algo,self.output,cpu_state,require_equal)
    def test_original_wrong_path_reproduces_native_error(self):
        with self.assertRaisesRegex(ValueError,'not in the logging directory'):self.algo.save(str(self.output/NAME))
        self.assertFalse((self.output/NAME).exists())
    def test_real_cpu_model_and_four_group_optimizer_roundtrip(self):
        result=self.run_export();self.assertTrue(result['native_reload_verified'])
        self.assertTrue(result['export_bytes_readback_verified']);self.assertTrue(result['native_path_guard_unchanged'])
        self.assertEqual((self.output/NAME).read_bytes(),(Path(self.algo.log_dir)/NAME).read_bytes())
        saved=torch.load(self.output/NAME,map_location='cpu',weights_only=True)
        self.assertEqual(len(saved['optimizer']['param_groups']),4)
        self.assertGreater(len(saved['optimizer']['state']),0)
    def test_native_save_failure_never_publishes_final(self):
        self.algo.fault='save'
        with self.assertRaises(ValueError):self.run_export()
        self.assertFalse((self.output/NAME).exists())
    def test_model_reload_mismatch_rejected(self):self.check_fault('model')
    def test_optimizer_reload_mismatch_rejected(self):self.check_fault('optimizer')
    def test_scalar_reload_mismatch_rejected(self):self.check_fault('scalar')
    def test_teacher_readiness_mismatch_rejected(self):self.check_fault('ready')
    def test_native_bytes_changed_on_reload_rejected(self):self.check_fault('bytes',ValueError)
    def check_fault(self,fault,error=RuntimeError):
        self.algo.fault=fault
        with self.assertRaises(error):self.run_export()
        self.assertFalse((self.output/NAME).exists())
    def test_existing_final_preserved(self):
        target=self.output/NAME;target.write_bytes(b'keep existing')
        with self.assertRaises(FileExistsError):self.run_export()
        self.assertEqual(target.read_bytes(),b'keep existing')
    def test_existing_native_preserved(self):
        target=Path(self.algo.log_dir)/NAME;target.write_bytes(b'keep existing')
        with self.assertRaises(FileExistsError):self.run_export()
        self.assertEqual(target.read_bytes(),b'keep existing')
    def test_existing_pending_preserved(self):
        target=self.output/(NAME+'.pending');target.write_bytes(b'keep pending')
        with self.assertRaises(FileExistsError):self.run_export()
        self.assertEqual(target.read_bytes(),b'keep pending')
    def test_directory_disagreement_rejected(self):
        self.algo.log_dir=str(self.output)
        with self.assertRaises(ValueError):self.run_export()
    def test_logdir_outside_output_rejected(self):
        self.algo.log_dir=str(self.output.parent);self.algo.logging_helper.log_dir=self.algo.log_dir
        with self.assertRaises(ValueError):self.run_export()
    def test_linked_logdir_rejected(self):
        link=self.output/'log-link';link.symlink_to(self.algo.log_dir,target_is_directory=True)
        self.algo.log_dir=str(link);self.algo.logging_helper.log_dir=str(link)
        with self.assertRaises(ValueError):self.run_export()

if __name__=='__main__':unittest.main(verbosity=2)