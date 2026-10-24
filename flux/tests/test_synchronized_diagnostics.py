"""CPU adapter contracts; these are not IsaacSim physics validation."""
import ast,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
from synchronized_diagnostics import SynchronizedDiagnostics
from test_root_state_proxy_contract import native_proxy_class

def native_push():
    path=Path('/workspace/task047/diagnostic-native/src/holosoma/holosoma/envs/wbt/wbt_manager.py')
    if not path.is_file():
        path=ROOT/'thirdparty/holosoma/src/holosoma/holosoma/envs/wbt/wbt_manager.py'
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!='5bd1b8626781773d5368a14fefb16e35e8ffdd85728bcd389d345da0fce87735':
        raise ValueError('Pinned WBT native push source changed')
    tree=ast.parse(data);fn=next(f for c in tree.body if isinstance(c,ast.ClassDef)
        for f in c.body if isinstance(f,ast.FunctionDef) and f.name=='_push_robots')
    ns={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    return ns['_push_robots']

class Tests(unittest.TestCase):
    def make(self,directory,decimation=2):
        n,j=32,29;names=['joint_'+str(i) for i in range(j)]
        for i,name in zip([4,5,10,11],['left_ankle_pitch_joint','left_ankle_roll_joint',
                                     'right_ankle_pitch_joint','right_ankle_roll_joint']):names[i]=name
        ids=list(reversed(range(j)));native=list(reversed(names))
        bodies=['left_ankle_roll_link','waist_pitch_link','right_ankle_roll_link']
        data=NS(joint_pos_target=torch.zeros(n,j),applied_torque=torch.arange(j).float().repeat(n,1),
                computed_torque=torch.arange(j).float().repeat(n,1)+1)
        sensor=NS(body_names=list(reversed(bodies)),data=NS(net_forces_w=torch.zeros(n,3,3)),
                  cfg=NS(update_period=.005))
        sensor.data.net_forces_w[:,0,2]=7.;sensor.data.net_forces_w[:,2,2]=3.
        backing=torch.zeros(n,13);backing[:,3]=1.
        sim=NS(dof_ids=ids,body_names=bodies,_robot=NS(joint_names=native,data=data),
            dof_pos=torch.zeros(n,j),dof_vel=torch.zeros(n,j),contact_sensor=sensor,
            robot_root_states=native_proxy_class()(backing),
            simulator_config=NS(sim=NS(control_decimation_steps=decimation,fps=decimation/.02)),
            _rigid_body_vel=torch.zeros(n,3,3),contact_forces=sensor.data.net_forces_w[:,[2,1,0]],
            torque_limits=torch.ones(j)*42,dof_vel_limits=torch.ones(j)*13.6,calls=0)
        action=NS(_requested_position_targets=torch.zeros(n,j),_applied_position_targets=torch.zeros(n,j))
        stats=NS(active=False,mask=torch.ones(n,dtype=torch.bool),terms={})
        m=NS(tracked_body_indexes=torch.tensor([1,2,0]),time_steps=torch.arange(n),
            motion_ids=torch.zeros(n,dtype=torch.long),joint_pos=torch.zeros(n,j),joint_vel=torch.zeros(n,j),
            body_pos_relative_w=torch.zeros(n,3,3),body_pos_w=torch.zeros(n,3,3),
            ref_pos_w=torch.zeros(n,3),robot_body_pos_w=torch.zeros(n,3,3),
            robot_body_lin_vel_w=torch.zeros(n,3,3),robot_ref_pos_w=torch.zeros(n,3))
        env=NS(num_envs=n,device='cpu',dt=.02,_update_tasks_before_termination=False,
            robot_config=NS(dof_names=names),simulator=sim,action_manager=NS(_term_instances={'action':action}),
            reset_buf=torch.zeros(n,dtype=torch.long),default_dof_pos=torch.zeros(n,j),default_dof_pos_base=torch.zeros(n,j),
            need_to_refresh_envs=torch.zeros(n,dtype=torch.bool),_max_push_vel=torch.ones(6)*.1,
            randomization_manager=None,push_robot_vel_buf=torch.zeros(n,6),record_push_robot_vel_buf=torch.zeros(n,6))
        def simulate():sim.calls+=1;sim.dof_pos.add_(.01)
        sim.simulate_at_each_physics_step=simulate
        sim.set_actor_root_state_tensor_robots=lambda *args:None
        def physics():
            for _ in range(decimation):sim.simulate_at_each_physics_step()
        env._physics_step=physics
        def check():
            stats.terms={'bad_tracking':env.reset_buf.bool().clone()}
        env._check_termination=check
        push=native_push();env._push_robots=lambda ids:push(env,ids)
        recorder=SynchronizedDiagnostics(env,m,stats,directory)
        self.addCleanup(recorder.close)
        return recorder,env,m,stats

    def rows(self,path,name):return [json.loads(x) for x in (Path(path)/('sync_'+name+'.jsonl')).read_text().splitlines()]
    def step(self,e):e._physics_step();e._check_termination()

    def test_full_horizon_counts_and_clock(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True
            for _ in range(128):self.step(e)
            coverage=c.finish()
            self.assertEqual(coverage['first_episode_env_steps'],4096)
            self.assertEqual(coverage['physics_env_substeps'],8192)
            self.assertEqual(e.simulator.calls,256)
            rows=self.rows(d,'physics_substeps')
            self.assertEqual(rows[0]['time_start_s'],0.)
            self.assertEqual(rows[-1]['time_end_s'],2.56)
            self.assertIsNone(rows[0]['measured_solver_joint_effort_nm'])

    def test_world_error_sign_and_norm(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True
            m.robot_body_pos_w[0,2]=torch.tensor([.3,.4,-.2]);self.step(e)
            row=self.rows(d,'control_steps')[0]
            self.assertEqual(row['body_error_xyz_m'][0][2],[.3,.4,-.2])
            self.assertAlmostEqual(row['body_error_norm_m'][0][2],(.29)**.5,places=6)

    def test_native_joint_and_reversed_contact_sensor_mapping(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;self.step(e)
            row=self.rows(d,'physics_substeps')[0]
            self.assertEqual(row['actuator_applied_torque_estimate_nm'][0],[24.,23.,18.,17.])
            self.assertEqual(row['foot_net_force_w_n'][0],[[0.,0.,3.],[0.,0.,7.]])
            self.assertEqual(len(row['q_before_rad'][0]),4)

    def test_warmup_not_recorded(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);self.step(e)
            self.assertEqual(c.steps,0);self.assertEqual(self.rows(d,'physics_substeps'),[])

    def test_failure_step_included_post_reset_env_excluded(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;e.reset_buf[3]=1;self.step(e)
            e.simulator.dof_pos.fill_(999.);s.mask[3]=False;e.reset_buf.zero_();self.step(e)
            rows=self.rows(d,'control_steps')
            self.assertIn(3,rows[0]['active_envs']);self.assertNotIn(3,rows[1]['active_envs'])
            self.assertAlmostEqual(rows[0]['q_rad'][3][0],.02,places=6)
            self.assertEqual(c.samples,63)

    def test_native_push_has_next_step_timestamp_and_actual_delta(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;self.step(e)
            e._push_robots(torch.tensor([0,1]));self.step(e)
            events=self.rows(d,'push_events');steps=self.rows(d,'control_steps')
            self.assertEqual(len(events),1);self.assertEqual(events[0]['after_control_step'],1)
            self.assertEqual(events[0]['affects_control_step'],2)
            self.assertEqual(steps[0]['input_push_event_ids'],[])
            self.assertEqual(steps[1]['input_push_event_ids'],[1])
            self.assertEqual(events[0]['root_velocity_delta'],events[0]['root_velocity_after'])

    def test_push_excludes_failed_first_episode_but_still_calls_native_all_ids(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;e.reset_buf[0]=1;self.step(e)
            e._push_robots(torch.tensor([0,1]))
            self.assertEqual(self.rows(d,'push_events')[0]['active_envs'],[1])
            self.assertTrue(e.need_to_refresh_envs[0])

    def test_no_rng_or_physics_mutation_by_recorder(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True
            rng=torch.get_rng_state().clone();self.step(e)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.allclose(e.simulator.dof_pos,torch.full((32,29),.02)))
            self.assertTrue(torch.equal(e.simulator._robot.data.joint_pos_target,torch.zeros(32,29)))

    def test_missing_torque_not_filled_with_zero_or_pd_estimate(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;del e.simulator._robot.data.applied_torque
            self.step(e)
            self.assertIsNone(self.rows(d,'physics_substeps')[0]['actuator_applied_torque_estimate_nm'])
            self.assertIn('applied_torque',c.missing)

    def test_reference_advance_detected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;e._physics_step();m.time_steps+=1
            with self.assertRaisesRegex(ValueError,'Reference advanced'):e._check_termination()

    def test_target_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;e.simulator._robot.data.joint_pos_target.fill_(1.)
            with self.assertRaisesRegex(ValueError,'Native target mismatch'):e._physics_step()

    def test_nonfinite_required_signal_fails_closed(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;e.simulator.contact_sensor.data.net_forces_w[0,0,0]=float('nan')
            with self.assertRaisesRegex(FloatingPointError,'Nonfinite'):e._physics_step()

    def test_incomplete_finish_is_not_success(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d)
            with self.assertRaisesRegex(ValueError,'Incomplete'):c.finish()
            self.assertFalse((Path(d)/'sync_coverage.json').exists())

    def test_output_is_flat_for_existing_bundle_packager(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;self.step(e)
            files=[p.name for p in Path(d).glob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.yaml')]
            self.assertEqual(len(files),5)
            self.assertIn('sync_physics_substeps.jsonl',files)

    def test_serialization_budget_fails_instead_of_silently_truncating(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);c.bytes_written=16*1024*1024
            with self.assertRaisesRegex(ValueError,'byte budget exceeded'):c.emit('control_steps',dict(step=1))
            self.assertEqual(self.rows(d,'control_steps'),[])

    def test_slow_sensor_is_rejected_without_changing_it(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);c.close();e.simulator.contact_sensor.cfg.update_period=.05
            with self.assertRaisesRegex(ValueError,'slower than physics'):
                SynchronizedDiagnostics(e,m,s,Path(d)/'other')
            self.assertEqual(e.simulator.contact_sensor.cfg.update_period,.05)

    def test_close_restores_native_wrappers(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            c,e,m,s=self.make(d);s.active=True;c.close();self.step(e)
            self.assertEqual(c.steps,0);self.assertEqual(e.simulator.calls,2)

    def test_existing_entrypoint_opt_in_and_no_parameter_changes(self):
        source=(ROOT/'flux/evaluate_protected_multiseed.py').read_text()
        self.assertIn('synchronized_diagnostics=False',source)
        self.assertIn("if args.synchronized_diagnostics:command.append('--synchronized-diagnostics')",source)
        self.assertIn('algo.actor_optimizer.step=forbid',source)
        tree=ast.parse((ROOT/'flux/synchronized_diagnostics.py').read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ['learn','backward','save_checkpoint','set_joint_position_target','set_joint_effort_target']:
            self.assertNotIn(name,attrs)

if __name__=='__main__':unittest.main()