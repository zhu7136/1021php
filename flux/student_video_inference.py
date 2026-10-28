"""Native scene observations -> frozen student only. No trainer or optimizer."""
import argparse,json,os,random,sys,time,types
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flux.student_video_contract import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 plan=json.loads(a.plan.read_text());validate_scope(plan['scope'])
 out=Path(plan['output']);out.mkdir(exist_ok=False)
 assert digest(plan['student'])==SOURCE_SHA
 import numpy as np
 import torch
 from wbt_training.utils.checkpoint import load_saved_experiment_config,apply_preprocess_hook
 from holosoma.utils.eval_utils import CheckpointConfig
 from wbt_training.utils.isaacsim_clock import use_php_isaacsim
 from holosoma.utils.sim_utils import setup_simulation_environment
 from wbt_training.utils.registry import bundle_pairs
 from flux.g1_eval283_protocol import scheduled_reset_source,relative_refresh_source
 from flux.video_preflight import encoder_preflight
 encoder_env,evidence=encoder_preflight(out/'encoder',os.environ);os.environ['PATH']=encoder_env['PATH']
 torch.manual_seed(42);random.seed(42);np.random.seed(42)
 ck=torch.load(plan['student'],map_location='cpu',weights_only=True);assert ck['iter']==976 and ck['num_teachers']==5
 lengths=[];names=[]
 for dataset,count in zip(plan['datasets'],COUNTS):
  pairs=bundle_pairs(Path(dataset));assert len(pairs)==count
  for motion,terrain,_ in pairs:
   assert terrain is not None
   with np.load(motion,allow_pickle=False) as d:
    assert float(d['fps'])==50
    lengths.append(len(d['joint_pos']));names.append(str(motion.relative_to(Path(dataset))))
 cfg,_=load_saved_experiment_config(CheckpointConfig(checkpoint=plan['student']))
 assert cfg.algo.config.empirical_normalization is False
 cfg=replace(cfg,training=replace(cfg.training,num_envs=1,headless=True,seed=42,checkpoint=None,
  teacher_checkpoint=None,wandb_path=None,export_onnx=False,
  registry_name=','.join(Path(x).resolve().as_uri() for x in plan['datasets']),
  preprocess_hook_kwargs=json.dumps(dict(add_onpath_obstacle=True,num_variants=3,obstacle_seed=42))),
  logger=replace(cfg.logger,mode='offline',base_dir=str(out),name='task216-five-skill-student-replay',
   video=replace(cfg.logger.video,enabled=True,interval=1000000,save_dir=str(out/'videos'),
    upload_to_wandb=False,show_command_overlay=False,record_env_id=0,use_recording_thread=False,width=640,height=360)))
 cfg=apply_preprocess_hook(use_php_isaacsim(cfg))
 term=cfg.command.setup_terms['motion_command'];mc=term.params['motion_config']
 overrides=dict(use_adaptive_timesteps_sampler=False,resample_on_motion_end=False,
  start_at_timestep_zero_prob=1.0,freeze_at_timestep_zero_prob=0.0)
 mc={**mc,**overrides} if isinstance(mc,dict) else replace(mc,**overrides)
 terms=dict(cfg.command.setup_terms);terms['motion_command']=replace(term,params={**term.params,'motion_config':mc})
 cfg=replace(cfg,command=replace(cfg.command,setup_terms=terms));cfg.save_config(str(out/'replay_config.yaml'))
 print(json.dumps(dict(event='student_video_simulator_initializing',training_updates=0,num_envs=1)),flush=True)
 env,device,app=setup_simulation_environment(cfg);assert str(device)=='cuda:0' and env.num_envs==1
 from holosoma.agents.modules.depth_backbone import DepthOnlyFCBackbone58x87Small
 from holosoma.agents.modules.student_teacher_modules import DepthStudentTeacher
 state=ck['model_state_dict'];mc=ck['experiment_config']['algo']['config']['module']
 policy=DepthStudentTeacher(num_actor_obs=108,num_teacher_obs=state['teachers.0.0.weight'].shape[1],
  num_actions=29,depth_backbone=DepthOnlyFCBackbone58x87Small(32),depth_output_dim=32,
  student_hidden_dims=mc['student_hidden_dims'],teacher_hidden_dims=mc['teacher_hidden_dims'],
  activation=mc['activation'],num_teachers=5).to(device)
 policy.load_state_dict(state,strict=True);policy.eval();policy.requires_grad_(False)
 for k,v in policy.state_dict().items():assert torch.equal(v.cpu(),state[k]),k
 def forbidden(*args,**kw):raise RuntimeError('Teacher actions/training forbidden in student replay')
 policy.teacher_act=forbidden
 cmd=env.command_manager.get_state('motion_command');base=cmd.motion
 ends=base.motion_ends.nonzero(as_tuple=False).flatten()+1
 starts=torch.cat([torch.zeros(1,device=device,dtype=torch.long),ends[:-1]])
 assert len(ends)==888 and np.array_equal((ends-starts).cpu().numpy(),np.tile(lengths,3))
 class SegmentedMotion:
  num_motions=888
  motion_start_idx=starts
  motion_end_idx=ends
  def __getattr__(self,k):return getattr(base,k)
 cmd.motion=SegmentedMotion()
 from holosoma.managers.command.terms import wbt
 namespace=dict(vars(wbt));source=Path(wbt.__file__).read_text()
 exec(compile(scheduled_reset_source(source),'scheduled_native_reset','exec'),namespace)
 exec(compile(relative_refresh_source(source),'native_relative_refresh','exec'),namespace)
 cmd.reset=types.MethodType(namespace['reset'],cmd)
 cmd.eval_motion_ids=torch.tensor([0],device=device,dtype=torch.long)
 env.set_is_evaluating();observed={};original=env._check_termination
 def capture():
  namespace['refresh_relative_targets'](cmd);original()
  observed.update(failed=bool(env.termination_manager.terminated[0]),timeout=bool(env.termination_manager.time_outs[0]))
 env._check_termination=capture
 recorder=env.simulator.video_recorder;assert recorder is not None
 # Keep the camera running over native failures/resets; retain their counts below.
 # Native failure thresholds and native environment resets are NOT suppressed.
 manual_stop=recorder.stop_recording
 recorder.stop_recording=lambda:None
 recorder.should_record_episode=lambda total:False
 results=[]
 for i,(skill,index) in enumerate(zip(SKILLS,SELECTED)):
  row=dict(skill=skill,original_clip_index=0,combined_motion_id=index,motion=names[index],
   training_updates=0,student_only=True,status='not_recorded',policy_steps=0,native_failure_steps=0,native_timeout_steps=0)
  results.append(row)
  left=plan['soft_deadline']-time.monotonic()
  if left<3:continue
  deadline=time.monotonic()+left/(5-i)
  cmd.eval_motion_ids[:]=index
  with torch.inference_mode():
   obs=env.reset_all();assert int(base.motion_idxs[cmd.time_steps][0])==i
   manual_stop();recorder.start_recording(i+1)
   # Stable unique filenames even when a native reset occurs within the clip.
   recorder._current_episode=i+1
   before=set((out/'videos').rglob('*.mp4'))
   fps=cfg.simulator.config.sim.fps/cfg.simulator.config.sim.control_decimation_steps
   for step in range(round(10*fps)):
    if time.monotonic()>=deadline:break
    actor=obs['actor_obs'];depth=obs['depth_camera'];depth=depth['depth_cam'] if isinstance(depth,dict) else depth
    assert actor.shape==(1,108) and depth.shape==(1,58,87)
    assert int(base.motion_idxs[cmd.time_steps][0])==i
    actions=policy.act_inference({'actor_obs':actor,'depth_obs':depth})
    assert torch.isfinite(actions).all()
    obs,_,_,_=env.step({'actions':actions});row['policy_steps']+=1
    row['native_failure_steps']+=int(observed.get('failed',False));row['native_timeout_steps']+=int(observed.get('timeout',False))
   manual_stop()
   created=sorted(set((out/'videos').rglob('*.mp4'))-before);assert len(created)==1
   target=out/'videos'/(str(i+1)+'-'+skill+'-student-task216.mp4');created[0].rename(target)
   row.update(status='recorded' if row['policy_steps']==round(10*fps) else 'budget_truncated',
    video=str(target.relative_to(out)),simulated_seconds=row['policy_steps']/fps,video_bytes=target.stat().st_size)
  (out/'clips.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(dict(event='student_video_clip_saved',**row)),flush=True)
 for k,v in policy.state_dict().items():assert torch.equal(v.cpu(),state[k]),k
 assert digest(plan['student'])==SOURCE_SHA
 result=dict(pipeline_passed=all(r['status'] in ('recorded','budget_truncated') and r['policy_steps']>0 for r in results),
  training_updates=0,student_only=True,strict_model_unchanged=True,source_checkpoint_unchanged=True,
  source_sha256=SOURCE_SHA,terrain_regenerated_not_historical=True,terrain_seed=42,terrain_variants=3,
  scope='One first original clip per skill, native resets retained, not a success-rate evaluation',clips=results)
 (out/'replay-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(event='student_video_replay_complete',**result)),flush=True)
 sys.stdout.flush();os._exit(0 if result['pipeline_passed'] else 1)
if __name__=='__main__':
 from flux.video_preflight import fail_closed_entrypoint
 fail_closed_entrypoint(main)