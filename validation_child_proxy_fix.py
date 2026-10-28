"""Actual IsaacSim load/reset and short implicit-PD probes; no learner exists."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from dataclasses import replace

def write_root_state_via_proxy(sim,ids,root_state):
 """Pinned IsaacSim expects RootStatesProxy, not a raw [N,13] tensor.

 MotionLoader and proxy indexing both expose xyzw; the proxy converts to
 IsaacSim wxyz. Preserve all position, orientation and velocity components.
 """
 import torch
 if root_state.ndim!=2 or tuple(root_state.shape)!=(ids.numel(),13):raise ValueError('Expected selected root states [len(ids),13]')
 if not bool(torch.isfinite(root_state).all()):raise ValueError('Nonfinite root state')
 sim.robot_root_states[ids]=root_state
 sim.set_actor_root_state_tensor_robots(ids,sim.robot_root_states)

def close_native_app(app,out,report):
 """Disable the pinned IsaacLab stop-event render loop; do not stub teardown."""
 from isaaclab.sim import SimulationContext
 context=SimulationContext.instance()
 report['simulator_close_started']=True
 report['usd_close_stage_unmodified']=True
 report['stop_render_wait_guard_available']=bool(context is not None and hasattr(context,'_disable_app_control_on_stop_handle'))
 if report['stop_render_wait_guard_available']:
  context._disable_app_control_on_stop_handle=True
  report['stop_render_wait_guard_enabled']=True
 (out/'validation-report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'event':'stage','stage':'simulator_close','stop_render_wait_guard_enabled':report.get('stop_render_wait_guard_enabled',False)}),flush=True)
 app.close(wait_for_replicator=False)
 report['simulator_close_returned']=True
 (out/'validation-report.json').write_text(json.dumps(report,indent=2))

def main():
 p=argparse.ArgumentParser()
 for n in ['payload','output','site','thread-id','turn-id']:p.add_argument('--'+n,required=True)
 a=p.parse_args();payload=Path(a.payload);out=Path(a.output);root=payload/'bundle'
 for q in [payload/'native-source',Path(a.site),root/'runtime/php',root/'runtime/holosoma/src/holosoma',root/'overlay',root,payload/'validation-code']:sys.path.insert(0,str(q))
 from runtime_versions import check_runtime_versions
 versions=check_runtime_versions();(out/'runtime-versions.json').write_text(json.dumps(versions))
 from portable_config import resolve
 from holosoma.config_types.experiment import ExperimentConfig
 from holosoma.config_types.logger import DisabledLoggerConfig
 raw=json.loads((payload/'validation-code/config.json').read_text())
 cfg=ExperimentConfig(**resolve(raw,root,out))
 assert cfg.training.num_envs==1 and cfg.algo.config.num_learning_iterations==0
 assert not cfg.training.checkpoint and not cfg.training.teacher_checkpoint and not cfg.training.registry_name
 assert len(cfg.robot.dof_names)==31 and cfg.robot.actions_dim==29
 cfg=replace(cfg,simulator=replace(cfg.simulator,_target_='wbt_training.simulators.isaacsim.IsaacSim'),logger=DisabledLoggerConfig(base_dir=str(out)))
 sys.argv=[sys.argv[0]]
 from holosoma.utils.sim_utils import setup_simulator_imports
 from holosoma.utils.eval_utils import init_sim_imports
 setup_simulator_imports(cfg);app=None;report={'task_id':os.environ['PHP_FLUX_TASK_ID'],'ppo_updates':0,'learner_constructed':False,'teacher_converged':False,'probe_count':0,'all240_physics_certified':False}
 try:
  app=init_sim_imports(cfg)
  import torch,numpy as np
  from holosoma.config_types.env import get_tyro_env_config
  from holosoma.utils.helpers import get_class
  env=get_class(cfg.env_class)(get_tyro_env_config(cfg),device='cuda:0');sim=env.simulator
  assert tuple(sim.dof_names)==tuple(cfg.robot.dof_names) and len(sim._body_list)==42
  report.update(native_environment_loaded=True,physical_joints=31,reference_bodies=42,cases=[])
  (out/'validation-report.json').write_text(json.dumps(report,indent=2))
  command=env.command_manager.get_state('motion_command')
  assert command is not None and hasattr(command,'motion') and hasattr(command,'time_steps')
  motion=command.motion
  assert sim.simulator_config.sim.fps==200 and sim.simulator_config.sim.control_decimation_steps==4
  cases=json.loads((payload/'validation-code/cases.json').read_text());assert len(cases)==4
  report.update(native_environment_loaded=True,physical_joints=31,reference_bodies=42,cases=[])
  ids=torch.tensor([0],device='cuda:0');origin=sim.scene.env_origins
  expected_bodies=list(np.load(root/'reference/validation-four.npz',allow_pickle=False)['body_names'])
  body_idx=[expected_bodies.index(n) for n in sim._body_list]
  limits=torch.tensor(cfg.robot.dof_effort_limit_list,device='cuda:0');footids=[sim._body_list.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
  for case in cases:
   cr={'dataset':case['dataset'],'clip':case['clip'],'probes':[]}
   for local in case['probe_frames']:
    report['active_probe']={'dataset':case['dataset'],'source_frame':local,'stage':'reset'}
    (out/'validation-report.json').write_text(json.dumps(report,indent=2))
    t=case['start_frame']+local;ix=torch.tensor([t],device='cuda:0');q=motion.get_joint_pos(ix);v=motion.get_joint_vel(ix)
    root_state=torch.cat([motion.get_body_pos_w(ix)[:,0]+origin,motion.get_body_quat_w(ix)[:,0],motion.get_body_lin_vel_w(ix)[:,0],motion.get_body_ang_vel_w(ix)[:,0]],-1)
    write_root_state_via_proxy(sim,ids,root_state)
    sim.set_dof_state_tensor_robots(ids,torch.stack([q,v],-1));sim.apply_position_targets_at_dof(q)
    sim.scene.write_data_to_sim();sim.sim.forward();sim.scene.update(dt=0.0);sim.refresh_sim_tensors()
    pose_error=float(torch.linalg.vector_norm(sim._rigid_body_pos-(motion._body_pos_w[ix][:,body_idx]+origin[:,None,:]),dim=-1).max())
    qerror=float((sim.dof_pos-q).abs().max())
    probe={'source_frame':local,'reset_joint_max_error_rad':qerror,'reset_body_fk_max_error_m':pose_error,'reset_mapping_passed':qerror<=1e-5 and pose_error<=.005,'steps':[]}
    report['active_probe']['stage']='short_physics'
    for k in range(min(25,case['frames']-local-1)):
     at=torch.tensor([t+k+1],device='cuda:0');target=motion.get_joint_pos(at)
     sim.apply_position_targets_at_dof(target)
     for _ in range(4):sim.simulate_at_each_physics_step()
     sim.refresh_sim_tensors()
     finite=bool(torch.isfinite(sim.dof_pos).all() and torch.isfinite(sim.dof_vel).all() and torch.isfinite(sim._rigid_body_pos).all())
     torque=sim._robot.data.applied_torque[:,sim.dof_ids]
     step={'frame':local+k+1,'finite':finite,'tracking_joint_max_error_rad':float((sim.dof_pos-target).abs().max()),'torque_limit_ratio_max':float((torque.abs()/limits).max()),'foot_contact_force_N_max':float(torch.linalg.vector_norm(sim.contact_forces[:,footids],dim=-1).max()),'all_contact_force_N_max':float(torch.linalg.vector_norm(sim.contact_forces,dim=-1).max()),'root_height_m':float(sim.robot_root_states[:,2].min())}
     probe['steps'].append(step)
     if not finite:break
    probe['short_physics_completed']=len(probe['steps'])==min(25,case['frames']-local-1) and all(x['finite'] for x in probe['steps'])
    cr['probes'].append(probe);report['probe_count']+=1
    (out/'validation-report.json').write_text(json.dumps(report|{'current_case':cr},indent=2))
   cr['load_reset_short_physics_passed']=all(p['reset_mapping_passed'] and p['short_physics_completed'] for p in cr['probes']);report['cases'].append(cr)
   (out/'validation-report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'event':'case_complete','dataset':case['dataset'],'passed':cr['load_reset_short_physics_passed']}),flush=True)
  report['native_validation_completed']=True;report['load_reset_short_physics_passed']=all(x['load_reset_short_physics_passed'] for x in report['cases'])
  report['contact_torque_recorded_not_certified']=True;report['collision_com_support_certified']=False
 except Exception as e:
  report.update(error_type=type(e).__name__,error_message=str(e)[:1500]);traceback.print_exc()
 finally:
  (out/'validation-report.json').write_text(json.dumps(report,indent=2))
  if app is not None:
   close_native_app(app,out,report)
 return 0 if report.get('load_reset_short_physics_passed') else 1
if __name__=='__main__':sys.exit(main())