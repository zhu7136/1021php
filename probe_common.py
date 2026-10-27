"""One bounded official-source transfer/image probe. No installation or simulator."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
import fetch_official_wheels as fetcher
PIN='6d0833e158ccb8a0fb3ac4776a467a1ac2a45ef72c99caa7e59da5ad38c96049'
def emit(event,**data):print(json.dumps(dict(event=event,**data),separators=(',',':')),flush=True)
def environment():
 names=['PATH','HOME','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','CONDA_PREFIX','VIRTUAL_ENV','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','LANG','LC_ALL','ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT']
 result={key:os.environ[key] for key in names if key in os.environ};result.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',WANDB_MODE='offline');return result
def run_json(code,vendor,deadline,limit):
 with fetcher.absolute_budget(min(limit,deadline-time.monotonic())):
  p=subprocess.run([vendor,'-c',code],capture_output=True,text=True,env=environment(),timeout=min(limit,deadline-time.monotonic()))
 if p.returncode:raise RuntimeError('image probe subprocess failed')
 lines=[line for line in p.stdout.splitlines() if line.startswith('{')]
 if not lines or len(lines[-1])>100000:raise ValueError('invalid image probe record')
 return json.loads(lines[-1])
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--task-id');parser.add_argument('--commit');args=parser.parse_args()
 if not args.execute:emit('hud04_delivery_probe_readonly',execution_requested=False,worker_seconds=300,download_seconds_max=180,new_installations=0,simulator_started=False,ppo_updates=0);return 0
 if not args.task_id or not re.fullmatch('TASK_[0-9]{8}_[0-9]+',args.task_id) or not args.commit or not re.fullmatch('[0-9a-f]{40}',args.commit):raise ValueError('task/source binding required')
 start=time.monotonic();deadline=start+300;root=Path(__file__).resolve().parent
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=environment(),timeout=10,text=True).strip()
 if head!=args.commit:raise ValueError('source checkout mismatch')
 state=dict(task_id=args.task_id,source_commit=head,worker_seconds_max=300,automatic_retries=0,download_passed=False,image_metadata_recorded=False,gpu_verified=False,installed=False,simulator_started=False,ppo_updates=0,verified_hud04_checkpoint=False)
 try:
  with fetcher.absolute_budget(max(1,deadline-time.monotonic())):
   vendor='/isaac-sim/python.sh'
   if not Path(vendor).is_file():raise ValueError('vendor interpreter missing')
   metadata="import importlib.metadata as m,json,sys,os,platform; d=dict(python_version=platform.python_version(),sys_platform=sys.platform,os_name=os.name,platform_machine=platform.machine(),python_implementation=platform.python_implementation(),packages=[dict(name=x.metadata.get('Name'),version=x.version) for x in m.distributions()]); print(json.dumps(d),flush=True)"
   state['image_metadata']=run_json(metadata,vendor,deadline,40);state['image_metadata_recorded']=True;emit('hud04_image_metadata',task_id=args.task_id,**state['image_metadata'])
   gpu="import torch,json; p=torch.cuda.get_device_properties(0); d=dict(torch=torch.__version__,gpu=p.name,memory_bytes=p.total_memory,compute_capability=[p.major,p.minor],gpu_count=torch.cuda.device_count()); print(json.dumps(d),flush=True)"
   state['gpu']=run_json(gpu,vendor,deadline,45);g=state['gpu'];state['gpu_verified']=g['torch']=='2.7.0+cu128' and g['gpu_count']==1 and '4090' in g['gpu'] and 23<=g['memory_bytes']/2**30<=25 and g['compute_capability']==[8,9];emit('hud04_gpu_metadata',task_id=args.task_id,**g)
   if not state['gpu_verified']:raise ValueError('target GPU/runtime differs')
   manifest=fetcher.load_manifest(root/'official-wheel-manifest.json',PIN)
   result=fetcher.fetch(manifest,root/('official-'+args.task_id),min(180,max(1,deadline-time.monotonic()-10)))
   result['worker_reachability_proven']=result['complete']
   state['download_result']=result;state['download_passed']=result['complete'];state['downloaded_files']=len(result['records']);state['downloaded_bytes']=result['total_bytes']
 except Exception as error:state['error_type']=type(error).__name__
 state['elapsed_seconds']=round(time.monotonic()-start,3);state['passed']=state['download_passed'] and state['image_metadata_recorded'] and state['gpu_verified'];emit('hud04_delivery_probe_result',**state);return 0 if state['passed'] else 1
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as error:emit('hud04_delivery_probe_failure',error_type=type(error).__name__,passed=False);sys.exit(1)