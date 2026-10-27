"""Bounded five-wheel probe; no dependency install, app, learner or checkpoint."""
import argparse,json,re,subprocess,sys,time
from pathlib import Path
import incremental
from fetch_official_wheels import absolute_budget
from probe_common import environment,run_json

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--task-id');parser.add_argument('--commit');args=parser.parse_args()
 if not args.execute:
  print(json.dumps(dict(execution_requested=False,network_requested=False,worker_seconds_max=240,download_seconds_max=180,files_requested=5,automatic_retries=0,installed=False,simulator_started=False,ppo_updates=0)),flush=True);return 0
 if not args.task_id or not re.fullmatch('TASK_[0-9]{8}_[0-9]+',args.task_id) or not args.commit or not re.fullmatch('[0-9a-f]{40}',args.commit):raise ValueError('task/commit binding required')
 start=time.monotonic();deadline=start+240;root=Path(__file__).resolve().parent;head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=environment(),timeout=10,text=True).strip()
 if head!=args.commit:raise ValueError('checkout mismatch')
 def emit(event,**data):print(json.dumps(dict(event=event,task_id=args.task_id,source_commit=head,**data),separators=(',',':')),flush=True)
 state=dict(task_id=args.task_id,source_commit=head,worker_seconds_max=240,download_seconds_max=180,automatic_retries=0,installed=False,simulator_started=False,ppo_updates=0,verified_hud04_checkpoint=False,full_runtime_closure_proven=False,whole_33_files_same_worker_verified=False,passed=False)
 try:
  with absolute_budget(max(0.001,deadline-time.monotonic())):
   rows=incremental.selected(root/'official-wheel-manifest.json');emit('hud04_incremental_scope',manifest_sha256=incremental.PIN,filenames=[r['filename'] for r in rows])
   code="""import importlib.metadata as m,importlib.util as u,json,platform,sys
names=['certifi','idna','requests','wandb','onnx','numpy','trimesh','torch']
rows=[]
for name in names:
 try:
  d=m.distribution(name);s=u.find_spec(name);rows.append(dict(name=name,version=d.version,metadata_root=str(d.locate_file('')),module_origin=s.origin if s else None,requires_dist=d.requires or []))
 except m.PackageNotFoundError:rows.append(dict(name=name,missing=True))
print(json.dumps(dict(python_version=platform.python_version(),sys_platform=sys.platform,platform_machine=platform.machine(),packages=rows)),flush=True)
"""
   image=run_json(code,'/isaac-sim/python.sh',deadline,35);state['image_metadata']=image;emit('hud04_incremental_effective_metadata',**image)
   if image['python_version']!='3.11.13' or image['sys_platform']!='linux' or image['platform_machine']!='x86_64':raise ValueError('unexpected target runtime')
   result=incremental.run(rows,root/('incremental-'+args.task_id),min(deadline-10,time.monotonic()+180),emit);state['download_result']=result;state['passed']=result['complete']
 except Exception as error:state['error_type']=type(error).__name__
 state['elapsed_seconds']=round(time.monotonic()-start,3);emit('hud04_incremental_final',**{k:v for k,v in state.items() if k not in ['task_id','source_commit']});return 0 if state['passed'] else 1
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as error:print(json.dumps(dict(event='hud04_incremental_entry_failure',error_type=type(error).__name__,passed=False)),flush=True);sys.exit(1)