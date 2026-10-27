"""Single image-native empty-physics baseline:450s worker,30s close,no retry."""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path

def environment():
 names=['PATH','HOME','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','CONDA_PREFIX','VIRTUAL_ENV','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','LANG','LC_ALL','ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT']
 env={k:os.environ[k] for k in names if k in os.environ};env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled');return env
def verify_sources(root):
 rows=json.loads((root/'source-manifest.json').read_text())['files']
 for row in rows:
  p=root/row['path']
  if '/' in row['path'] or p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('source mismatch')
 return hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest()
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=2)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=2)
def collect(out):
 result={}
 for name in ['runtime.json','probe.json','child-error.json','close-settings.json','close-returned.json','lifecycle-returned.json']:
  p=out/name
  if p.exists():
   if p.stat().st_size>30000:raise ValueError('oversized receipt')
   result[name]=json.loads(p.read_text())
 p=out/'close-stack.log'
 if p.exists():result['close_stack_tail']=p.read_text(errors='replace')[-18000:]
 return result
def run_child(argv,out,env,deadline,emit):
 rows=[];seen=0;closing=None;reason='exited';p=None
 with (out/'child.log').open('xb') as log:
  p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   while True:
    journal=out/'events.jsonl'
    if journal.exists():
     for line in journal.read_text().splitlines()[seen:]:
      try:row=json.loads(line)
      except ValueError:break
      seen+=1;rows.append(row);emit('minimal_stage',**row)
      if row['stage']=='app_close_started':closing=row['monotonic']
    if p.poll() is not None:break
    if closing is not None and time.monotonic()-closing>30:reason='close_deadline';stop(p);break
    if time.monotonic()>deadline:reason='worker_deadline';stop(p);break
    time.sleep(.2)
  finally:stop(p)
 raw=(out/'child.log').read_bytes();receipts=collect(out);settings=receipts.get('close-settings.json',{})
 passed=bool(p.returncode==0 and reason=='exited' and receipts.get('probe.json',{}).get('physics_steps')==8 and receipts.get('close-returned.json',{}).get('returned') is True and receipts.get('lifecycle-returned.json',{}).get('returned') is True and settings.get('readback') is False and settings.get('usd_method_unchanged') is True and settings.get('stop_disable_after') is True and settings.get('python_profiler_installed') is False)
 return dict(passed=passed,child_exit_code=p.returncode,supervisor_reason=reason,events=rows,receipts=receipts,child_log_bytes=len(raw),child_log_sha256=hashlib.sha256(raw).hexdigest(),child_log_tail=raw[-14000:].decode(errors='replace'))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
 for key in ['task-id','commit','thread-id','turn-id']:parser.add_argument('--'+key)
 a=parser.parse_args()
 if not a.execute:
  print(json.dumps(dict(execution_requested=False,worker_seconds_max=450,close_seconds_max=30,physics_steps_planned=8,robot_environments=0,ppo_updates=0,automatic_retries=0,new_installations=0)));return 0
 if not a.task_id or not re.fullmatch('TASK_[0-9]{8}_[0-9]+',a.task_id) or not a.commit or not re.fullmatch('[0-9a-f]{40}',a.commit) or not a.thread_id or not a.turn_id:raise ValueError('task/commit/provenance required')
 started=time.monotonic();deadline=started+450;root=Path(__file__).resolve().parent;repo=root.parents[1]
 def emit(event,**data):print(json.dumps(dict(event=event,task_id=a.task_id,source_commit=a.commit,**data),separators=(',',':')),flush=True)
 state=dict(passed=False,worker_seconds_max=450,close_seconds_max=30,physics_steps_planned=8,robot_environments=0,ppo_updates=0,automatic_retries=0,new_installations=0)
 try:
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,env=environment(),text=True,timeout=10).strip()==a.commit,'checkout mismatch'
  state['source_manifest_sha256']=verify_sources(root)
  out=root/('probe-'+a.task_id);out.mkdir(exist_ok=False)
  emit('minimal_child_launch',worker_seconds_max=450,remaining_seconds=round(deadline-time.monotonic(),3),new_installations=0)
  state.update(run_child(['/isaac-sim/python.sh',str(root/'child.py'),'--execute','--output',str(out),'--thread-id',a.thread_id,'--turn-id',a.turn_id],out,environment(),deadline-12,emit))
 except Exception as error:state.update(passed=False,error_type=type(error).__name__,error_message=str(error)[:600],frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-8:]])
 state['elapsed_seconds']=round(time.monotonic()-started,3);emit('minimal_final',**state);return 0 if state['passed'] else 1
if __name__=='__main__':sys.exit(main())