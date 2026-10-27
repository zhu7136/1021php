"""Single600s teacher worker;default readonly;private task parameters only for scoped input."""
import argparse,base64,hashlib,json,os,signal,subprocess,sys,time,zipfile
from pathlib import Path
def main():
 start=time.monotonic();deadline=start+600
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--task-id');p.add_argument('--commit');p.add_argument('--thread-id');p.add_argument('--turn-id');p.add_argument('--source-capability');a=p.parse_args()
 if not a.execute:print(json.dumps(dict(execution_requested=False,gpu_started=False,worker_seconds=600,max_updates=1000,gate_updates=20,automatic_retries=0)));return 0
 if not all([a.task_id,a.commit,a.thread_id,a.turn_id,a.source_capability]):raise ValueError('task/source/provenance/input binding required')
 root=Path(__file__).resolve().parent;repo=root.parents[1];strict=repo/'flux/g1_strict';sys.path[:0]=[str(root),str(strict),str(repo)]
 from fetch_official_wheels import absolute_budget
 from dependency_install import workflow,command
 from probe_common import environment
 from scoped_transport import transfer
 env=environment();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,env=env,text=True,timeout=10).strip()
 if head!=a.commit:raise ValueError('commit mismatch')
 manifest=json.loads((root/'source-manifest.json').read_text())
 for row in manifest['files']:
  path=root/row['path']
  if path.is_symlink() or path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('source manifest mismatch')
 out=root/('result-'+a.task_id);out.mkdir(exist_ok=False);child=None;reason='not_started';put=None
 def emit(event,**kw):print(json.dumps(dict(event=event,task_id=a.task_id,**kw)),flush=True)
 state=dict(task_id=a.task_id,source_commit=head,worker_seconds_max=600,max_updates=1000,gate_updates=20,automatic_retries=0,passed=False)
 try:
  with absolute_budget(max(.001,deadline-time.monotonic()-15)):
   controls=json.loads(base64.urlsafe_b64decode(a.source_capability));a.source_capability=None
   if set(controls)!={'get','put'}:raise ValueError('only single-object input/output capabilities allowed')
   cap=controls['get'];put=controls['put'];controls=None
   checkpoint=out/'source-checkpoint.pt'
   state['input_transfer']=transfer(cap,checkpoint,method='GET',size=10346299,sha256='cb98b0762fbf3b478211bcb9131d53d0330c3459fa7dba04c7874205c1875399',absolute_deadline=min(deadline-300,time.monotonic()+45));cap=None;checkpoint.chmod(0o444)
   emit('walk_source_verified',sha256=state['input_transfer']['sha256'])
   # Both workflow and child share this absolute deadline;reserve save/close/pack time.
   state['dependencies']=workflow(strict,a.task_id,min(deadline-180,time.monotonic()+300),emit)
   inputs=out/'inputs';prepared=command(['/isaac-sim/python.sh',str(repo/'flux/prepare_inputs.py'),'--output',str(inputs)],out/'prepare.log',deadline-150,60)
   if prepared['returncode']!=0:raise ValueError('input reconstruction failed')
   env.update(PYTHONPATH=os.pathsep.join([str(strict/('isolation-'+a.task_id)/'site'),str(root),str(strict),str(repo),str(repo/'thirdparty/holosoma/src/holosoma')]),WANDB_MODE='offline',WANDB_ENTITY='zhul136-university-of-science-and-technology-of-china',WANDB_PROJECT='HuD04_WBT')
   if deadline-time.monotonic()<140:raise TimeoutError('not enough learning/save/close budget')
   argv=['/isaac-sim/python.sh',str(root/'child.py'),'--output',str(out),'--data',str(inputs/'motions-locomotion'),'--checkpoint',str(checkpoint),'--thread-id',a.thread_id,'--turn-id',a.turn_id]
   with (out/'child.log').open('xb') as log:
    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=repo);emit('walk_child_started',remaining_seconds=round(deadline-time.monotonic(),3));reason='running'
    while child.poll() is None:
     now=time.monotonic();events=[]
     if (out/'events.jsonl').exists():events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines() if x]
     if now>=deadline-130:(out/'STOP_AFTER_UPDATE').touch(exist_ok=True)
     closing=[x for x in events if x['stage']=='app_close_started']
     if (closing and now-closing[-1]['monotonic']>=30) or now>=deadline-75:
      reason='close_timeout' if closing else 'child_deadline';break
     time.sleep(.5)
    if child.poll() is not None:reason='exited'
    else:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=3)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
   state.update(child_exit_code=child.returncode,supervisor_reason=reason)
   if (out/'learning-result.json').exists():state['learning']=json.loads((out/'learning-result.json').read_text())
   state['native_close_returned']=(out/'close-returned.json').exists();state['lifecycle_returned']=(out/'lifecycle-returned.json').exists()
   state['passed']=child.returncode==0 and state['native_close_returned'] and state['lifecycle_returned'] and state.get('learning',{}).get('gate_passed') is True
 except BaseException as error:
  state['error_type']=type(error).__name__
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
 state['elapsed_seconds']=round(time.monotonic()-start,3)
 (out/'worker-status.json').write_text(json.dumps(state,indent=2))
 # Platform discovery is additional;single-object PUT below is the explicit result channel.
 for path in out.rglob('*.pt'):
  if time.monotonic()>=deadline-4:break
  if path!=out/'source-checkpoint.pt':emit('walk_checkpoint',file=str(path.relative_to(repo)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 archive=out/'walk-evidence.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for path in out.rglob('*'):
   if time.monotonic()>=deadline-45:raise TimeoutError('result pack cutoff')
   wanted=path.suffix in ['.json','.jsonl','.log','.yaml'] or path.name in ['model_gate20.pt','model_smoke_final.pt']
   if path.is_file() and wanted and path.stat().st_size<20*1024*1024:z.write(path,path.relative_to(out))
 if put is not None:
  receipt=transfer(put,archive,method='PUT',size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),absolute_deadline=deadline-3)
  emit('walk_result_upload',**receipt)
 emit('walk_final',**{k:v for k,v in state.items() if k!='task_id'});return 0 if state['passed'] else 1
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException as e:
  if isinstance(e,SystemExit):raise
  print(json.dumps(dict(event='walk_entry_failure',error_type=type(e).__name__)),flush=True);sys.exit(1)