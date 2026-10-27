"""Bounded create_empty.py-derived baseline; no robot, PHP, Holosoma or overlay.
Reference: IsaacLab v2.3.1 create_empty.py (BSD-3-Clause).
Changes:8 steps,headless CUDA,strict close settings/receipts/deadlines.
"""
import argparse,faulthandler,hashlib,inspect,json,platform,sys,time,traceback
from pathlib import Path

def lifecycle(out,provenance):
 def event(stage,**data):
  row=dict(stage=stage,monotonic=time.monotonic(),**data)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  print('MINIMAL_EVENT '+json.dumps(row),flush=True)
 def receipt(name,**data):
  with (out/name).open('x') as f:json.dump(data,f);f.flush()
 app=None
 try:
  from isaaclab.app import AppLauncher
  event('app_launch_started')
  launcher=AppLauncher(headless=True,device='cuda:0');app=launcher.app
  event('app_launch_returned')
  import carb.settings
  import omni.usd
  usd_cls=type(omni.usd.get_context());usd_original=usd_cls.close_stage
  import torch
  from isaaclab.sim import SimulationCfg,SimulationContext
  gpu=torch.cuda.get_device_properties(0)
  source_hashes={k:hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() for k,cls in [('app_launcher',AppLauncher),('simulation_context',SimulationContext)]}
  receipt('runtime.json',python=platform.python_version(),torch=torch.__version__,gpu=gpu.name,memory_bytes=gpu.total_memory,gpu_count=torch.cuda.device_count(),isaaclab_release=Path('/workspace/isaaclab/VERSION').read_text().strip(),isaacsim_release=Path('/isaac-sim/VERSION').read_text().strip(),source_hashes=source_hashes)
  if platform.python_version()!='3.11.13' or torch.__version__!='2.7.0+cu128' or torch.cuda.device_count()!=1 or '4090' not in gpu.name:raise ValueError('target runtime/GPU mismatch')
  event('context_build_started');sim=SimulationContext(SimulationCfg(dt=.01,device='cuda:0'));event('context_build_returned')
  sim.reset();event('reset_returned')
  for _ in range(8):sim.step()
  event('short_control_returned',physics_steps=8)
  forbidden=[n for n in sys.modules if n=='holosoma' or n.startswith('holosoma.') or n=='wbt_training' or n.startswith('wbt_training.')]
  if forbidden:raise ValueError('unexpected project framework imported')
  receipt('probe.json',physics_steps=8,robot_loaded=False,robot_environments=0,php_imported=False,holosoma_imported=False,overlays_installed=False,ppo_updates=0,_wb_agent=provenance)
 except BaseException as error:
  receipt('child-error.json',error_type=type(error).__name__,message=str(error)[:1200],frames=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-10:]])
  raise
 finally:
  if app is not None:
   import carb.settings
   import omni.usd
   from isaaclab.sim import SimulationContext
   settings=carb.settings.get_settings();before=settings.get('/app/fastShutdown')
   if type(before) is not bool:raise ValueError('unreadable fastShutdown')
   settings.set_bool('/app/fastShutdown',False);readback=settings.get('/app/fastShutdown')
   if readback is not False:raise ValueError('fastShutdown must be false')
   context=SimulationContext.instance()
   if context is None:raise ValueError('missing live SimulationContext')
   stop_before=context._disable_app_control_on_stop_handle
   context._disable_app_control_on_stop_handle=True
   # Retain r53 STOP setting; leave official USD close and app.close untouched.
   unchanged=type(omni.usd.get_context()).close_stage is usd_original
   receipt('close-settings.json',before=before,readback=readback,usd_method_unchanged=unchanged,stop_before=stop_before,stop_disable_after=context._disable_app_control_on_stop_handle,python_profiler_installed=sys.getprofile() is not None)
   if not unchanged or sys.getprofile() is not None:raise ValueError('USD changed or unexpected profiler')
   event('app_close_started',close_budget_seconds=30)
   with (out/'close-stack.log').open('x') as trace:
    faulthandler.dump_traceback_later(5,repeat=True,file=trace)
    try:app.close(wait_for_replicator=False)
    finally:faulthandler.cancel_dump_traceback_later()
   receipt('close-returned.json',returned=True);event('app_close_returned')
 receipt('lifecycle-returned.json',returned=True,_wb_agent=provenance)
 return 0

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path);parser.add_argument('--thread-id');parser.add_argument('--turn-id');a=parser.parse_args()
 if not a.execute:
  print(json.dumps(dict(execution_requested=False,simulator_started=False,physics_steps_planned=8,ppo_updates=0)));return 0
 if not a.output or not a.output.is_dir() or not a.thread_id or not a.turn_id:raise ValueError('fresh supervisor output/provenance required')
 sys.argv=[sys.argv[0]]
 return lifecycle(a.output,dict(thread_id=a.thread_id,turn_id=a.turn_id))
if __name__=='__main__':sys.exit(main())