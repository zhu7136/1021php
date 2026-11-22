"""Run one exact frozen low-climb teacher clip; never instantiate an algorithm."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from dataclasses import replace

def main():
 p=argparse.ArgumentParser();p.add_argument('--payload',required=True);p.add_argument('--output',required=True);p.add_argument('--site',required=True);p.add_argument('--checkpoint',required=True)
 a=p.parse_args();payload=Path(a.payload);out=Path(a.output);root=payload/'bundle';out.mkdir(exist_ok=True)
 entry=Path(__file__).resolve().parent;plan=json.loads((entry/'PLAN.json').read_text())
 for path in [payload/'native-source',Path(a.site),root/'runtime/php',root/'runtime/holosoma/src/holosoma',root/'overlay',root,payload/'teacher-code',entry]:sys.path.insert(0,str(path))
 import torch
 from torch import nn
 from runtime_versions import check_runtime_versions
 (out/'runtime-versions.json').write_text(json.dumps(check_runtime_versions(),indent=2))
 from portable_config import resolve
 from holosoma.config_types.experiment import ExperimentConfig
 from holosoma.config_types.logger import DisabledLoggerConfig
 cfg=ExperimentConfig(**resolve(json.loads((payload/'teacher-code/config.json').read_text()),root,out))
 cfg=replace(cfg,logger=DisabledLoggerConfig(base_dir=str(out)))
 checkpoint=Path(a.checkpoint);digest=lambda:hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 assert digest()==plan['checkpoint']['sha256']
 c=torch.load(checkpoint,map_location='cpu',weights_only=True);assert c['iter']==649 and c['actor_obs_normalizer_state_dict'] is None
 actor=nn.Sequential(nn.Linear(449,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,29)).eval()
 prefix='actor_module.module.';actor.load_state_dict({k[len(prefix):]:v for k,v in c['actor_model_state_dict'].items() if k.startswith(prefix)},strict=True)
 actor.requires_grad_(False);initial={k:v.detach().clone() for k,v in actor.state_dict().items()}
 assert cfg.training.num_envs==1 and cfg.training.headless and not cfg.training.checkpoint
 sys.argv=[sys.argv[0]]
 from holosoma.utils.sim_utils import setup_simulation_environment
 import close_audit
 app=None;env=None;report=dict(close_started=False,close_returned=False,comparison_returned=False,training_updates=0)
 try:
  env,device,app=setup_simulation_environment(cfg)
  close_audit.configure(app,out,report)
  from native_binding import EpisodeStopped
  from speed10_audit import install_native_audit
  speed_flush=install_native_audit(env,out)
  env.set_is_evaluating();sim=env.simulator;assert abs(env.dt-.02)<1e-9 and abs(env.sim_dt-.005)<1e-9
  assert len(sim.dof_names)==31 and tuple(sim.dof_names)==tuple(cfg.robot.dof_names)
  assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
  # Match the CPU replay's translated world, without a settling or zero-action step.
  sim.scene.env_origins.zero_();ids=torch.arange(1,device=device);m=env.command_manager.get_state('motion_command')
  m.init_buffers();env.reset_envs_idx(ids);env._refresh_envs_after_reset(ids)
  assert int(m.time_steps[0])==0
  # Compute initial relative-body targets at frame zero without advancing physics.
  # The upstream step() otherwise uses the root rather than the configured waist
  # anchor when episode_length==0; avoid that training-reset-only workaround.
  env.episode_length_buf[:]=1;m.time_steps[:]=-1;m.step();env.episode_length_buf[:]=0
  env._pre_compute_observations_callback()
  scanner=sim.scene.sensors['height_scanner'];scanner.update(0.,force_recompute=True)
  env._compute_observations();env._post_compute_observations_callback();env._clip_observations()
  expected=torch.cat([m.root_pos_w,m.root_quat_w,m.root_lin_vel_w,m.root_ang_vel_w],dim=-1)
  root_error=float((sim.robot_root_states[:]-expected).abs().max());joint_error=float((sim.dof_pos-m.joint_pos).abs().max());velocity_error=float((sim.dof_vel-m.joint_vel).abs().max())
  init=dict(initial_root_max_abs_error=root_error,initial_joint_max_abs_error=joint_error,initial_joint_velocity_max_abs_error=velocity_error,no_initial_physics_steps=True,reference_frame=int(m.time_steps[0]),native_joint_names=list(sim.dof_names),native_body_names=list(sim._body_list),native_hard_joint_limits=sim.hard_dof_pos_limits.detach().cpu().tolist(),physics_dt=float(env.sim_dt),policy_dt=float(env.dt),device_name=torch.cuda.get_device_name(0),checkpoint_sha256=digest(),checkpoint_iteration=649,reference_frames=800,observation_shape=list(env.obs_buf_dict['actor_obs'].shape),training_updates=0,randomization_disabled=True,initial_anchor_for_relative_target='waist_pitch_link',_wb_agent=plan['provenance'])
  (out/'initial-state-proof.json').write_text(json.dumps(init,indent=2));assert max(root_error,joint_error,velocity_error)<1e-4 and init['observation_shape']==[1,449]
  actor=actor.to(device);env.comparison_stream=(out/'trajectory.jsonl').open('w');env.comparison_armed=True
  env.snapshot('initial_before_any_physics');obs_rows=[];outcome='full_reference_clip_reached'
  with torch.inference_mode():
   for step in range(799):
    obs=env.obs_buf_dict['actor_obs'];assert torch.isfinite(obs).all()
    act=actor(obs);assert torch.isfinite(act).all();obs_rows.append(dict(policy_step=step,reference_frame=int(m.time_steps[0]),observation=obs[0].cpu().tolist(),action=act[0].cpu().tolist()))
    try:env.step(dict(actions=act))
    except EpisodeStopped as stopped:outcome=str(stopped);break
  speed_flush(0)
  env.comparison_armed=False;env.comparison_stream.close()
  (out/'policy-input-output.json').write_text(json.dumps(obs_rows,allow_nan=False))
  assert digest()==plan['checkpoint']['sha256']
  assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in actor.state_dict().items())
  result=dict(outcome=outcome,full_reference_clip_reached=outcome=='full_reference_clip_reached',policy_steps=int(env.episode_length_buf[0]),simulated_seconds=float(env.episode_length_buf[0])*env.dt,last_snapshot=env.comparison_last,source_checkpoint_sha256=digest(),weights_unchanged=True,training_updates=0,automatic_resets=0,no_initial_physics_steps=True,skills_certified=False,initial_state_proof=init,comparison_scope='one frozen training clip, native PhysX versus prior MuJoCo screen; not multi-seed skill certification')
  (out/'native-result.json').write_text(json.dumps(result,allow_nan=False,indent=2));print('NATIVE_FROZEN_RESULT '+json.dumps({k:result[k] for k in ['outcome','policy_steps','simulated_seconds','weights_unchanged','training_updates']}),flush=True)
  report['comparison_returned']=True
 finally:
  if app is not None:close_audit.close(app,out,report)
  (out/'child-exit-state.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()