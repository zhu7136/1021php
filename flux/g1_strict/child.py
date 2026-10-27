"""G1 strict-close control with an import-only check of the exact execution configuration."""
import argparse,faulthandler,json,sys,time,traceback
from pathlib import Path

def record(out,name,**data):
 with (out/name).open('x') as f:json.dump(data,f,indent=2);f.flush()
def native_close(app,train,out,event):
 import carb.settings
 import omni.usd
 from isaaclab.sim import SimulationContext
 from helper_variant import make_variant
 settings=carb.settings.get_settings();before=settings.get('/app/fastShutdown')
 if type(before) is not bool:raise ValueError('unreadable fastShutdown')
 settings.set_bool('/app/fastShutdown',False);after=settings.get('/app/fastShutdown')
 record(out,'close-settings.json',before=before,requested=False,readback=after)
 if after is not False:raise ValueError('fastShutdown not false')
 usd_class=type(omni.usd.get_context());original_usd=usd_class.close_stage
 context=SimulationContext.instance()
 record(out,'helper-before.json',context_present=context is not None,stop_disable_before=getattr(context,'_disable_app_control_on_stop_handle',None),python_profiler_installed=False)
 def preclose():
  current=SimulationContext.instance();unchanged=usd_class.close_stage is original_usd
  flag=getattr(current,'_disable_app_control_on_stop_handle',None)
  record(out,'helper-at-close.json',context_present=current is not None,usd_method_unchanged=unchanged,stop_disable_after=flag)
  if not unchanged or (current is not None and flag is not True):raise ValueError('unexpected helper mutation or STOP flag')
 variant,proof=make_variant(train.close_simulation_app,preclose)
 record(out,'helper-variant.json',**proof)
 event('app_close_started',close_budget_seconds=30,fast_shutdown=False,python_profiler_installed=False)
 with (out/'close-stack.log').open('x') as trace:
  faulthandler.dump_traceback_later(5,repeat=True,file=trace)
  try:variant(app)
  finally:faulthandler.cancel_dump_traceback_later()
 record(out,'close-returned.json',returned=True,fast_shutdown=False)
 event('app_close_returned')
def lifecycle(config,out,provenance):
 from holosoma.utils.sim_utils import setup_simulator_imports
 setup_simulator_imports(config)
 import holosoma.train_agent as train
 app=None;env=None
 def event(stage,**kw):
  row=dict(stage=stage,monotonic=time.monotonic(),**kw)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  print('G1_STRICT_EVENT '+json.dumps(row),flush=True)
 try:
  event('app_launch_started');app=train.init_sim_imports(config);event('app_launch_returned')
  import torch
  from holosoma.utils.common import seeding
  seeding(config.training.seed,torch_deterministic=config.training.torch_deterministic)
  def finite(value):
   if isinstance(value,torch.Tensor):
    if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite simulation tensor')
    return 1
   if isinstance(value,dict):return sum(finite(v) for v in value.values())
   if isinstance(value,(tuple,list)):return sum(finite(v) for v in value)
   return 0
  if config.training.preprocess_hook:
   from holosoma.managers.utils import resolve_callable
   hook=resolve_callable(config.training.preprocess_hook,context='preprocess_hook')
   event('task_preprocess_started');config=hook(config,**json.loads(config.training.preprocess_hook_kwargs or '{}'));event('task_preprocess_returned')
  event('env_build_started');env=train.get_class(config.env_class)(train.get_tyro_env_config(config),device='cuda:0')
  if env.num_envs!=2 or env.dim_actions!=29 or env.num_dof!=29:raise ValueError('G1 env/action/dof mismatch')
  event('env_build_returned',num_envs=2,action_dim=29,physical_dofs=29)
  obs=env.reset_all()
  if finite(obs)<1:raise ValueError('reset produced no tensors')
  event('reset_returned')
  with torch.no_grad():
   for _ in range(8):
    result=env.step({'actions':torch.zeros((2,29),device='cuda:0')})
    if finite(result)<2 or len(result)!=4 or result[1].numel()!=2:raise ValueError('invalid short-control result')
  event('short_control_returned',explicit_control_steps=8,including_reset_control_steps=9)
  record(out,'probe.json',robot='g1',num_envs=2,physical_dofs=29,action_dim=29,explicit_control_steps=8,finite_observations_and_rewards=True,ppo_updates=0,learner_instantiated=False,checkpoint_loaded=False,task_quality_evaluated=False,_wb_agent=provenance)
 except BaseException as error:
  record(out,'child-error.json',error_type=type(error).__name__,message=str(error)[:1200],frames=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-10:]])
  raise
 finally:
  try:
   if env is not None:
    event('env_cleanup_started')
    if hasattr(env,'close'):env.close()
    elif hasattr(env,'simulator') and hasattr(env.simulator,'close'):env.simulator.close()
    else:raise ValueError('missing environment cleanup')
    event('env_cleanup_returned')
  finally:
   if app is not None:native_close(app,train,out,event)
 record(out,'lifecycle-returned.json',returned=True,ppo_updates=0,_wb_agent=provenance)
 return 0
def make_config(repo,data):
 sys.path.insert(0,str(repo))
 from wbt_training.config_values.experiment import g1_wbt_terrain_ref_noscale
 from wbt_training.utils.isaacsim_clock import use_php_isaacsim
 from probe_config import build_probe_config
 return use_php_isaacsim(build_probe_config(g1_wbt_terrain_ref_noscale,data.resolve().as_uri()))
def main():
 p=argparse.ArgumentParser()
 for name in ['repo','data','output']:p.add_argument('--'+name,type=Path,required=True)
 for name in ['thread-id','turn-id']:p.add_argument('--'+name,required=True)
 p.add_argument('--config-check-only',action='store_true')
 a=p.parse_args();repo=a.repo.resolve();sys.path.insert(0,str(repo))
 cfg=make_config(repo,a.data)
 if a.config_check_only:
  assert cfg.logger.type=='disabled' and cfg.training.num_envs==2 and cfg.training.wandb_path is None
  assert cfg.training.checkpoint is None and cfg.training.teacher_checkpoint is None and not cfg.logger.video.enabled
  assert cfg.simulator._target_=='wbt_training.simulators.isaacsim.IsaacSim'
  print(json.dumps(dict(config_check_passed=True,preset='g1_wbt_terrain_ref_noscale',logger_type=cfg.logger.type,simulator_target=cfg.simulator._target_,num_envs=cfg.training.num_envs,preprocess_hook=cfg.training.preprocess_hook,simulator_started=False,ppo_updates=0,output_created=a.output.exists())),flush=True)
  return 0
 from flux.preflight import runtime_checks
 record(a.output,'runtime.json',**runtime_checks(a.data))
 sys.argv=[sys.argv[0]]
 return lifecycle(cfg,a.output,dict(thread_id=a.thread_id,turn_id=a.turn_id))
if __name__=='__main__':sys.exit(main())