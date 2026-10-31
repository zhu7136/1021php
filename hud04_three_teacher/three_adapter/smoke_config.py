"""Pure config assembly from an authenticated Hu-D04 teacher template.

No G1 preset import, no simulator startup, no execution/submission side effects.
Runtime paths must point to parent-verified immutable inputs.
"""
import copy
from pathlib import Path
from .contract import SKILLS
from .command_contract import STUDENT_TERM_WIDTHS

ROUTE_BODIES=('left_ankle_roll_link','right_ankle_roll_link','left_wrist_roll_link','right_wrist_roll_link')

def build_smoke_dict(teacher_template,*,bundle,smoke,teachers,output,mesh_vertices,mesh_faces):
 if len(teachers)!=3 or len(set(map(str,teachers)))!=3:raise ValueError('Three ordered distinct teacher inputs required')
 c=copy.deepcopy(teacher_template);bundle=Path(bundle);smoke=Path(smoke);output=Path(output)
 robot=c['robot'];tracked=c['command']['setup_terms']['motion_command']['params']['motion_config']['body_names_to_track']
 if robot['actions_dim']!=29 or len(robot['dof_names'])!=31 or not set(ROUTE_BODIES)<=set(tracked):
  raise ValueError('Not the authenticated Hu-D04 physical31/policy29 contract')
 if not {'head_yaw_joint','head_pitch_joint'}<=set(robot['dof_names']):raise ValueError('Head joints missing')
 base_actor=copy.deepcopy(c['observation']['groups']['actor_obs'])
 if base_actor['history_length']!=1 or 'which_motion' in base_actor['terms']:raise ValueError('Unexpected teacher observations')
 if base_actor['terms']['dof_pos']['func']!='wbt_training.hud04_state:policy_dof_pos':raise ValueError('Expected Hu-D04 policy-state adapter')
 student=copy.deepcopy(base_actor)
 student['terms']={k:copy.deepcopy(base_actor['terms'][k]) for k in ('actions','base_ang_vel','dof_pos','dof_vel')}
 student['terms']['robot_anchor_projected_gravity']={'func':'holosoma.managers.observation.terms.wbt:robot_anchor_projected_gravity','params':{},'scale':1.,'noise':.05,'clip':None}
 student['terms']['velocity_command']={'func':'three_adapter.command_contract:desired_velocity_command','params':{},'scale':1.,'noise':0.,'clip':None}
 if set(student['terms'])!=set(STUDENT_TERM_WIDTHS):raise ValueError('Student input terms changed')
 teacher=copy.deepcopy(base_actor);teacher['enable_noise']=False
 thresholds=c['termination']['terms']['bad_tracking']['params']
 teacher['terms']['which_motion']={'func':'wbt_training.config_values.motion_command:which_motion',
  'params':{**{k:thresholds[k] for k in ('bad_ref_pos_threshold','bad_ref_ori_threshold','bad_motion_body_pos_threshold')},
   'bad_motion_body_pos_body_names':list(ROUTE_BODIES)},'scale':1.,'noise':0.,'clip':None}
 depth={'terms':{'depth_cam':{'func':'holosoma.managers.observation.terms.depth:WarpDepthImageObsTerm',
  'params':{'camera_sensor_cfg':'three_adapter.chest_camera.HuD04ChestDepthConfig','command_name':'motion_command',
   'latency_frame':[3,4],'resize':[58,87],'buffer_len':6,'enable_holes':True,
   'terrain_mesh_vertices':mesh_vertices,'terrain_mesh_faces':mesh_faces},'scale':1.,'noise':0.,'clip':None}},
  'concatenate':False,'enable_noise':False,'history_length':1}
 c['observation']['groups']={'actor_obs':student,'teacher_obs':teacher,'depth_camera':depth}
 c['algo']={'_target_':'three_adapter.native_algo.HuD04SmokeDistillation','_recursive_':False,
  'config':{'module':{'student_hidden_dims':[2048,1024,512,256,128],'teacher_hidden_dims':[512,256,128],
   'activation':'ELU','depth_backbone':'holosoma.agents.modules.depth_backbone.DepthOnlyFCBackbone58x87Small','depth_output_dim':32},
   'num_learning_iterations':5,'num_steps_per_env':24,'num_learning_epochs':2,'num_mini_batches':2,
   'learning_rate':3e-4,'max_grad_norm':1.,'save_interval':100,'init_noise_std':.001,
   'empirical_normalization':False,'distill_loss_type':'mse','distillation_warmup_steps':1}}
 c['training'].update(num_envs=6,headless=True,multigpu=False,checkpoint=None,teacher_checkpoint=','.join(map(str,teachers)),
  registry_name=None,wandb_path=None,preprocess_hook=None,preprocess_hook_kwargs='{}',export_onnx=False,
  project='HuD04_WBT',name='hud04-three-teacher-student-unapproved-smoke')
 # Resolve explicitly; the original templates have $BUNDLE paths, not portable
 # runtime files. No automatic search or G1 asset fallback.
 asset=robot['asset'];asset.update(asset_root=str(bundle/'asset'),mesh_root=str(bundle/'asset/meshes'),
  urdf_file=str(bundle/'asset/urdf/HU_D04_01.urdf'),usd_file=str(bundle/'asset/usd/HU_D04_01.usd'))
 c['terrain']['terrain_term']['obj_file_path']=str(smoke/'terrain.obj')
 for group in ('setup_terms','reset_terms','step_terms'):
  c['command'][group]['motion_command']['func']='three_adapter.native_command:HuD04BalancedMotionCommand'
 c['command']['setup_terms']['motion_command']['params']['motion_config']['motion_file']=str(smoke/'motion.npz')
 c['simulator']['_target_']='wbt_training.simulators.isaacsim.IsaacSim'
 c['logger']={'type':'disabled','base_dir':str(output)}
 return c

def check_live_observations(manager):
 """Measure actual native term widths and offsets BEFORE optimizer creation."""
 import torch
 widths={};offsets={};observations={}
 for group,expected in [('actor_obs',96),('teacher_obs',450)]:
  cfg=manager.cfg.groups[group]
  if not cfg.concatenate or cfg.history_length!=1:raise ValueError('Flat single-frame groups required')
  offset=0;widths[group]={};offsets[group]={}
  for name in sorted(cfg.terms):
   value=manager._compute_term(group,name,cfg.terms[name])
   if value.ndim!=2 or not torch.isfinite(value).all():raise ValueError('Invalid live term '+name)
   width=value.shape[1];widths[group][name]=width;offsets[group][name]=offset;offset+=width
  if offset!=expected:raise ValueError(f'Live {group} width {offset}, expected {expected}')
  observations[group]=manager.compute_group(group,modify_history=False)
 if widths['actor_obs']!=STUDENT_TERM_WIDTHS:raise ValueError('Student term widths changed')
 if widths['teacher_obs']['which_motion']!=1:raise ValueError('Teacher routing width changed')
 depth=manager.compute_group('depth_camera',modify_history=False)['depth_cam']
 if tuple(depth.shape[1:])!=(58,87) or not torch.isfinite(depth).all():raise ValueError('Invalid live depth input')
 return {'widths':widths,'offsets':offsets,'depth_shape':list(depth.shape[1:]),'live_widths_verified':True}