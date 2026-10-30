"""CPU-only native-source lifecycle regression with fake physics/capture.

Uses actual pinned manager + task reset/observation scheduling + depth buffer
methods. Physics, cameras, randomization and command state are test doubles.
No native simulator imports, initialization, training, or network access.
"""
import ast
from collections import deque
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import zipfile
import torch

from hud04_depth_reset_guard import PendingDepthResetMixin

ROOT=Path('/workspace/hud_input_audit/core/bundle/runtime/holosoma/src/holosoma/holosoma')
DEPTH_SHA='6d1483e1381e8ccb66aa6f23c236da2678783023a9b0f9d71af25942f133502c'
MANAGER_SHA='3c5ac22ba33894ce42a4ac12cb067dfd59da33b52f8fca5fc30f16586f371470'
TASK_SHA='4e66b8aea909c4ddc3d2cf344a33d5883c7e04edd1d0724be2b9e01b88ffe115'

def source_slice(source, digest, class_name, method_names, namespace):
    if hashlib.sha256(source).hexdigest()!=digest:
        raise ValueError('Pinned source changed; refuse silent test substitution')
    tree=ast.parse(source)
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in method_names]
    if {n.name for n in methods}!=set(method_names):
        raise ValueError('Missing native method')
    selected=ast.ClassDef(name=class_name,bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'hash_pinned_native_slice','exec'),namespace)
    return namespace[class_name]

DepthMethods=source_slice((ROOT/'managers/observation/terms/depth.py').read_bytes(),DEPTH_SHA,
    'WarpDepthImageObsTerm',['reset','_ensure_command_resolved','__call__'],{'torch':torch})
Manager=source_slice((ROOT/'managers/observation/manager.py').read_bytes(),MANAGER_SHA,
    'ObservationManager',['__init__','_initialize_groups','_accepts_modify_history','compute',
      'compute_group','_compute_term','reset','get_obs_dims'],
    {'torch':torch,'inspect':inspect,'deque':deque})
with zipfile.ZipFile('/workspace/hud_control/private-input.zip') as z:
    member=next(n for n in z.namelist() if n.endswith('/envs/base_task/base_task.py'))
    task_source=z.read(member)
TaskMethods=source_slice(task_source,TASK_SHA,'BaseTask',
    ['reset_all','reset_envs_idx','_post_physics_step','_compute_observations',
     '_compute_final_observations','_ensure_long_tensor','_get_envs_to_refresh'],
    {'torch':torch,'Phase':NS(EPISODE_START='start',EPISODE_END='end',FRAME_END='frame')})

class FakeTermBase:
    pass

class NativeDepth(DepthMethods,FakeTermBase):
    def __init__(self,cfg,env):
        self.num_envs=env.num_envs;self.device='cpu';self.buffer_len=6
        self.depth_buffer=torch.zeros(env.num_envs,6,58,87)
        self._cached_output=torch.zeros(env.num_envs,58,87)
        self.reset_episodes=torch.zeros(env.num_envs,dtype=torch.bool)
        self.command=None;self._prev_time_step=None;self._command_name='motion_command'
        self.latency_frame_range=env.latency_range;self.latency_frame=env.latency
        self.env=env;self.captures=0
    def _get_depth_images(self,env):
        self.captures+=1;env.events.append(('capture',self.reset_episodes.tolist()))
        return torch.full((env.num_envs,58,87),-.4+self.captures*.01)
    def _process_depth_images(self,images,env_ids):
        return images

class GuardedDepth(PendingDepthResetMixin,NativeDepth):
    pass

class FakeCommand:
    def __init__(self,n):self.time_steps=torch.zeros(n,dtype=torch.long)
    def reset(self,ids):self.time_steps[ids]=0
    def step(self):self.time_steps+=1

class FakeTask(TaskMethods):
    def __init__(self,*,guarded,n=1,latency=4,latency_range=None):
        self.num_envs=n;self.device='cpu';self.dim_actions=29;self.events=[];self.actions=[]
        self.latency=latency;self.latency_range=latency_range;self.is_evaluating=True
        self.command=FakeCommand(n);self.command_manager=NS(get_state=lambda _:self.command,
            reset=self.command.reset)
        self.simulator=NS(hooks=NS(emit=lambda *args:None),robot_root_states=torch.zeros(n,13),
            dof_state=torch.zeros(n,31,2),set_actor_root_state_tensor_robots=lambda *args:None,
            set_dof_state_tensor_robots=lambda *args:None)
        self.randomization_manager=None;self.action_manager=None;self.curriculum_manager=None
        self.termination_manager=None;self.reset_manager=NS(reset_scene=lambda ids:None)
        self._pending_episode_lengths=torch.zeros(n,dtype=torch.long)
        self._pending_episode_update_mask=torch.zeros(n,dtype=torch.bool)
        self.episode_length_buf=torch.zeros(n,dtype=torch.long)
        self.reset_buf=torch.zeros(n,dtype=torch.long);self.to_reset=torch.empty(0,dtype=torch.long)
        self.extras={};self.log_dict={};self.viewer=None;self._update_tasks_before_termination=False
        term=NS(func='test_depth',params={},noise=0.,scale=1.,clip=None)
        group=NS(terms={'depth_cam':term},concatenate=False,enable_noise=False,history_length=1)
        cfg=NS(groups={'depth_camera':group})
        Manager._initialize_groups.__globals__['resolve_callable']=lambda *args,**kwargs:GuardedDepth if guarded else NativeDepth
        Manager._initialize_groups.__globals__['ObservationTermBase']=FakeTermBase
        self.observation_manager=Manager(cfg,self,'cpu')
        self.depth=self.observation_manager._term_instances['depth_camera']['depth_cam']
    def _reset_envs_idx_impl(self,ids,*args):
        self.events.append(('reset_states',ids.tolist()))
    def _refresh_sim_tensors(self):pass
    def _update_counters_each_step(self):pass
    def _pre_compute_observations_callback(self):pass
    def _post_compute_observations_callback(self):pass
    def _clip_observations(self):pass
    def _compute_reward(self):pass
    def _update_log_dict(self):pass
    def _check_termination(self):
        self.reset_buf.zero_();self.reset_buf[self.to_reset]=1
    def _update_tasks_callback(self):self.command.step()
    def _store_final_observations(self,ids,obs):self.final_observation=obs
    def step(self,state):
        self.actions.append(state['actions'].clone());self._post_physics_step()
        return self.obs_buf_dict,torch.zeros(self.num_envs),self.reset_buf.clone(),self.extras

def image(obs):return obs['depth_camera']['depth_cam']

class DepthResetGuardTests(unittest.TestCase):
    def test_native_full_lifecycle_reproduces_initial_zero_frames(self):
        env=FakeTask(guarded=False,latency=4)
        observations=[image(env.reset_all()).clone()]
        for _ in range(4):observations.append(image(env.step({'actions':torch.zeros(1,29)})[0]).clone())
        self.assertEqual([bool(x.eq(0).all()) for x in observations],[True,True,True,True,False])
        self.assertEqual(env.events[0],('reset_states',[0]))
        self.assertEqual(env.events[1],('reset_states',[]))
        self.assertEqual(env.events[2],('capture',[False]))

    def test_guarded_full_lifecycle_first_policy_input_nonzero(self):
        env=FakeTask(guarded=True,latency=4)
        first=image(env.reset_all())
        self.assertTrue(first.ne(0).all())
        self.assertTrue(torch.allclose(first,torch.full_like(first,-.39)))
        self.assertEqual(env.events[2],('capture',[True]))
        self.assertFalse(env.depth.reset_episodes.any())

    def test_native_empty_reset_erases_pending_flag(self):
        env=FakeTask(guarded=False);env.depth.reset(torch.tensor([0]));env.depth.reset(torch.empty(0,dtype=torch.long))
        self.assertFalse(env.depth.reset_episodes.any())

    def test_guarded_empty_reset_keeps_pending_flag(self):
        env=FakeTask(guarded=True);env.depth.reset(torch.tensor([0]));env.depth.reset(torch.empty(0,dtype=torch.long))
        self.assertTrue(env.depth.reset_episodes.all())

    def test_pending_resets_accumulate_for_distinct_environments(self):
        env=FakeTask(guarded=True,n=3);env.depth.reset(torch.tensor([0]));env.depth.reset(torch.tensor([2]))
        self.assertEqual(env.depth.reset_episodes.tolist(),[True,False,True])

    def test_reset_all_then_partial_does_not_discard_other_pending(self):
        env=FakeTask(guarded=True,n=3);env.depth.reset();env.depth.reset(torch.tensor([1]))
        self.assertTrue(env.depth.reset_episodes.all())

    def test_nonmutating_dimension_queries_do_not_capture(self):
        env=FakeTask(guarded=True);env.depth.reset()
        self.assertEqual(env.observation_manager.get_obs_dims(),{'depth_camera':{'depth_cam':(58,87)}})
        self.assertEqual(env.depth.captures,0);self.assertTrue(env.depth.reset_episodes.all())

    def test_nonmutating_reads_preserve_pending_and_last_frame(self):
        env=FakeTask(guarded=True);before=env.depth._cached_output.clone();env.depth.reset()
        obs=env.observation_manager.compute(modify_history=False)
        self.assertTrue(torch.equal(image(obs),before));self.assertEqual(env.depth.captures,0)
        self.assertTrue(env.depth.reset_episodes.all())

    def test_normal_latency_four_is_not_removed(self):
        env=FakeTask(guarded=True,latency=4);outputs=[image(env.reset_all()).clone()]
        for _ in range(5):outputs.append(image(env.step({'actions':torch.zeros(1,29)})[0]).clone())
        values=[round(float(x[0,0,0]),2) for x in outputs]
        self.assertEqual(values,[-.39,-.39,-.39,-.39,-.39,-.38])

    def test_latency_three_is_not_removed(self):
        env=FakeTask(guarded=True,latency=3);outputs=[image(env.reset_all()).clone()]
        for _ in range(4):outputs.append(image(env.step({'actions':torch.zeros(1,29)})[0]).clone())
        self.assertEqual([round(float(x[0,0,0]),2) for x in outputs],[-.39,-.39,-.39,-.39,-.38])

    def test_random_latency_three_four_no_zero_startup(self):
        with torch.random.fork_rng():
            torch.manual_seed(734);env=FakeTask(guarded=True,n=5,latency=None,latency_range=[3,4])
            self.assertTrue(image(env.reset_all()).ne(0).all())
            for _ in range(5):self.assertTrue(image(env.step({'actions':torch.zeros(5,29)})[0]).ne(0).all())

    def test_command_time_backward_refills_buffer(self):
        env=FakeTask(guarded=True);env.reset_all();env.command.time_steps.fill_(10)
        env.observation_manager.compute();env.command.time_steps.zero_()
        current=image(env.observation_manager.compute())
        self.assertTrue(torch.allclose(current,torch.full_like(current,-.37)))

    def test_partial_reset_fills_only_selected_environment(self):
        env=FakeTask(guarded=True,n=2);env.depth.depth_buffer.fill_(.2)
        env.depth.reset(torch.tensor([1]));env.depth.reset(torch.empty(0,dtype=torch.long))
        current=image(env.observation_manager.compute())
        self.assertTrue(current[0].eq(.2).all());self.assertTrue(torch.allclose(current[1],torch.full_like(current[1],-.39)))

    def test_exception_before_capture_keeps_pending_reset(self):
        env=FakeTask(guarded=True);env.depth.reset()
        def fail(_):raise RuntimeError('synthetic camera failure')
        env.depth._get_depth_images=fail
        with self.assertRaises(RuntimeError):env.observation_manager.compute()
        self.assertTrue(env.depth.reset_episodes.all())

    def test_empty_reset_after_consumption_does_not_create_reset(self):
        env=FakeTask(guarded=True);env.reset_all();env.depth.reset(torch.empty(0,dtype=torch.long))
        self.assertFalse(env.depth.reset_episodes.any())

    def test_no_implicit_installation_actions_or_termination_changes(self):
        baseline=FakeTask(guarded=False);guarded=FakeTask(guarded=True)
        baseline.reset_all();guarded.reset_all()
        action=torch.arange(29,dtype=torch.float32).reshape(1,-1)/100
        a=baseline.step({'actions':action});b=guarded.step({'actions':action})
        self.assertIs(type(baseline.depth),NativeDepth);self.assertIs(type(guarded.depth),GuardedDepth)
        self.assertTrue(torch.equal(a[2],b[2]));self.assertEqual(len(baseline.actions),len(guarded.actions))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(baseline.actions,guarded.actions)))
        self.assertTrue(torch.equal(baseline.command.time_steps,guarded.command.time_steps))


if __name__=='__main__':unittest.main()