"""Local formal worker DRAFT. No scheduler, downloader or submission API.

CLI defaults to request validation, never execution. An approved runner must
call execute_approved() after independent human budget approval and gate
clearance. CPU tests inject subprocess fixtures; they are not GPU evidence.
"""
import argparse,hashlib,json,math,os,signal,subprocess,sys,tarfile,threading,time
from pathlib import Path
from .formal_core import validate_plan
from .worker_budget import isolated_child_env
from .runtime_imports import runtime_pythonpath

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def write_json(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(tmp,path)

def validate_request(request):
 if request.get('schema')!='hud04-formal-worker-request-v1':raise ValueError('Unknown request schema')
 p=validate_plan(request['plan'])
 if p['execution_approved'] is not True:raise PermissionError('Formal execution still gated; preparation is NOT compute approval')
 if p['execution_gates']:
  # A separately approved engineering trial may MEASURE these unresolved
  # conditions, never mark them passed or authorize long training/resume.
  risk=request.get('engineering_validation')
  known={'native_shutdown_exit_zero_unresolved','native_cross_process_resume_unverified','formal_worker_and_collector_native_gpu_unverified'}
  continuation=request.get('continuation_validation')
  bounded_resume=(continuation=={'purpose':'authorized_bounded_optimizer_continuation','max_additional_updates':1000,'start_completed_updates':100,'source_checkpoint_sha256':'e295f4acb723bb93e0cb8946b49c81e242dcaf73a84c5dc981486f1725ce1b15','shutdown_fault_accepted_not_hidden':True,'trajectory_exact_resume':False}
    and (p['additional_updates'],p['num_envs'],p['worker_seconds_max'],p['save_interval'])==(1000,10,1800,250)
    and request.get('resume') is not None and request.get('resume_sha256')==continuation['source_checkpoint_sha256'])
  if bounded_resume:
   if risk!={'purpose':'measure_unresolved_training_pipeline','unresolved_gates':p['execution_gates'],'long_training_authorized':True,'resume_authorized':True} or not set(p['execution_gates']).issubset(known):raise PermissionError('Bounded continuation must retain unresolved gates')
  else:
   if not isinstance(risk,dict) or risk!={'purpose':'measure_unresolved_training_pipeline','unresolved_gates':p['execution_gates'],'long_training_authorized':False,'resume_authorized':False} or not set(p['execution_gates']).issubset(known):raise PermissionError('Formal execution still gated')
   if (p['additional_updates'],p['num_envs'],p['worker_seconds_max'],p['save_interval'])!=(100,10,1800,25) or request.get('resume') is not None:raise PermissionError('Only an explicitly scoped fresh trial or bounded continuation may measure unresolved gates')
 seconds=p['worker_seconds_max']
 if type(seconds) is not int or not 60<=seconds<=86400:raise ValueError('Explicit 60..86400 second total worker budget required')
 approval=request['approval']
 if approval.get('confirmed_by_user') is not True or approval.get('purpose')!='formal_pilot' or approval.get('max_tasks')!=1:raise PermissionError('Fresh, single-task pilot approval required')
 if approval.get('worker_seconds_max')!=seconds or approval.get('additional_updates')!=p['additional_updates'] or approval.get('num_envs')!=p['num_envs']:raise PermissionError('Approval scope mismatch')
 if request.get('automatic_retries')!=0:raise ValueError('No retries')
 if not all(isinstance(p['_wb_agent'].get(k),str) and p['_wb_agent'][k] for k in ('thread_id','turn_id')):raise ValueError('Missing turn provenance')
 for key in ('runtime_root','bundle','data_root','teacher_root','output','vendor_python'):
  if not Path(request[key]).is_absolute():raise ValueError('Explicit absolute runner path required: '+key)
 if request.get('resume') is not None:
  if not Path(request['resume']).is_absolute() or len(request.get('resume_sha256',''))!=64:raise ValueError('Explicit hash-pinned resume required')
 if request.get('runtime_manifest_sha256') is None:raise ValueError('Pinned runtime/code input manifest required')
 return request

class AbsoluteBudget:
 """Kill only subprocess groups created here; timeout is failure, never success."""
 def __init__(self,seconds,*,started=None):
  if isinstance(seconds,bool) or not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or seconds<=0:raise ValueError('Finite budget required')
  self.started=time.monotonic() if started is None else started;self.deadline=self.started+seconds;self.seconds=seconds
  self.child=None;self.lock=threading.Lock();self.cancel=threading.Event();self.used=set();self.thread=None
 def remaining(self,fraction=1.):return max(0.,self.started+self.seconds*fraction-time.monotonic())
 def require(self,fraction=1.):
  if self.remaining(fraction)<=0:raise TimeoutError('Absolute worker deadline')
 def start_guard(self):
  if self.thread is not None:raise RuntimeError('Cannot restart watchdog')
  def guard():
   if not self.cancel.wait(self.remaining()):
    with self.lock:
     if self.child is not None and self.child.poll() is None:
      try:os.killpg(self.child.pid,signal.SIGKILL)
      except ProcessLookupError:pass
    os._exit(124) # emergency FAILURE only; never mask a native exit
  self.thread=threading.Thread(target=guard,daemon=True);self.thread.start()
 def close(self):self.cancel.set()
 def run(self,name,command,*,log,env,cwd,fraction):
  if name in self.used:raise RuntimeError('Stage retry prohibited')
  self.used.add(name);self.require(fraction);expired=False
  with Path(log).open('xb') as stream:
   with self.lock:
    if self.child is not None:raise RuntimeError('Concurrent native subprocess prohibited')
    child=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);self.child=child
   try:
    try:code=child.wait(timeout=self.remaining(fraction))
    except subprocess.TimeoutExpired:
     expired=True
     try:os.killpg(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     code=child.wait(timeout=max(.01,min(2.,self.remaining())))
   finally:
    with self.lock:self.child=None
  receipt={'stage':name,'returncode':code,'timed_out':expired,'elapsed_seconds':time.monotonic()-self.started}
  if expired:raise TimeoutError(json.dumps(receipt))
  return receipt

def verify_runtime_inputs(request):
 root=Path(request['runtime_root']).resolve();manifest=root/'formal-runtime-manifest.json'
 if sha(manifest)!=request['runtime_manifest_sha256']:raise ValueError('Runtime/code manifest hash mismatch')
 if request.get('resume') is not None:
  checkpoint=Path(request['resume']).resolve()
  if not checkpoint.is_relative_to(root) or sha(checkpoint)!=request['resume_sha256']:raise ValueError('Resume file identity/root mismatch')
 entries=json.loads(manifest.read_text())['files']
 if not entries:raise ValueError('Empty runtime manifest')
 for name,expected in entries.items():
  path=(root/name).resolve()
  if not path.is_relative_to(root) or not path.is_file() or sha(path)!=expected:raise ValueError('Runtime input mismatch')
 # Roots themselves cannot escape the authenticated runtime tree.
 for key in ('bundle','data_root','teacher_root'):
  if not Path(request[key]).resolve().is_relative_to(root):raise ValueError('Input root escapes pinned runtime')
 # Require every code/bundle/teacher/data input file to be listed. Exclude only
 # generated caches. A generated dependency/site tree also needs authentication.
 for folder in ('student_adapter','bundle','dependency-site',Path(request['data_root']).relative_to(root),Path(request['teacher_root']).relative_to(root)):
  directory=root/folder
  if not directory.is_dir():raise ValueError('Missing pinned runtime directory')
  for path in directory.rglob('*'):
   if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
    if path.is_symlink() or str(path.relative_to(root)) not in entries:raise ValueError('Unpinned or symlinked runtime file')

def validate_native_result(out,request,child_receipt):
 if child_receipt['returncode']!=0 or child_receipt['timed_out']:raise ValueError('Native subprocess did not exit normally')
 p=request['plan'];lifecycle=json.loads((out/'formal-lifecycle.json').read_text());result=json.loads((out/'formal-native-result.json').read_text())
 if not all(lifecycle.get(k) is True for k in ('train_returned','close_returned','child_returned','fast_shutdown_disabled_verified')):raise ValueError('Native normal-cleanup evidence missing')
 if result.get('additional_updates_completed')!=p['additional_updates'] or result.get('_wb_agent')!=p['_wb_agent'] or lifecycle.get('_wb_agent')!=p['_wb_agent']:raise ValueError('Native scope/provenance mismatch')
 if result.get('continuation_contract_sha256') is None or result.get('teachers_frozen_verified') is not True:raise ValueError('Missing native training checks')
 if result.get('checkpoint_model_readback_verified') is not True:raise ValueError('Final checkpoint readback missing')
 from .formal_core import continuation_contract,contract_digest
 if result['continuation_contract_sha256']!=contract_digest(continuation_contract(p)):raise ValueError('Wrong checkpoint contract')
 checkpoint=(out/result['checkpoint']).resolve()
 if not checkpoint.is_relative_to(out.resolve()) or not checkpoint.is_file() or sha(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Native checkpoint hash mismatch')
 total=result.get('completed_updates')
 if type(total) is not int or total<p['additional_updates']:raise ValueError('Missing completed-update counter')
 progress=list(out.rglob('formal_progress.jsonl'))
 if len(progress)!=1:raise ValueError('Expected exactly one formal progress stream')
 rows=[json.loads(line) for line in progress[0].read_text().splitlines() if line.strip()]
 if [r['additional_updates_completed'] for r in rows]!=list(range(1,p['additional_updates']+1)):raise ValueError('Incomplete/duplicate additional update indices')
 if [r['completed_updates'] for r in rows]!=list(range(total-p['additional_updates']+1,total+1)):raise ValueError('Ambiguous total update indices')
 return result

def seal_outputs(out,budget):
 """Manifest + tar of outputs, no parent environment or signed transfer URLs."""
 budget.require(.96);members={}
 for path in sorted(out.rglob('*')):
  if path.is_symlink():raise ValueError('Symlinked result forbidden')
  if path.is_file():budget.require(.96);members[str(path.relative_to(out))]={'sha256':sha(path),'bytes':path.stat().st_size}
 write_json(out/'result-manifest.json',{'schema':'hud04-formal-results-v1','files':members})
 archive=out.parent/(out.name+'.tar.gz')
 with tarfile.open(archive,'x:gz') as tar:
  for name in (*members,'result-manifest.json'):budget.require(.96);tar.add(out/name,arcname=name,recursive=False)
 return {'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size}

def execute_approved(request,*,publisher=None):
 """Future approved runner only. publisher(archive, receipt, remaining_seconds)
 must upload to the SAME private project and independently read back identity.
 No success without publication receipt. No automatic retry or scheduler call.
 """
 entered=time.monotonic();validate_request(request)
 out=Path(request['output']);out.mkdir(parents=True,exist_ok=False)
 budget=AbsoluteBudget(request['plan']['worker_seconds_max'],started=entered);budget.start_guard()
 state={'worker_passed':False,'training_executed':False,'skills_evaluated':False,'stage':'input_validation','automatic_retries':0,'_wb_agent':request['plan']['_wb_agent']}
 receipt=None
 try:
  verify_runtime_inputs(request);budget.require(.25)
  if request.get('resume') and sha(request['resume'])!=request['resume_sha256']:raise ValueError('Resume checksum mismatch')
  root=Path(request['runtime_root']);plan_path=out/'formal-plan.json';write_json(plan_path,request['plan'])
  child_env=isolated_child_env(os.environ,pythonpath=os.pathsep.join(map(str,runtime_pythonpath(root))),output=out,
   asset_root=Path(request['bundle'])/'asset',train_deadline=budget.started+budget.seconds*.75,task_id=request['task_id'])
  for key in ('ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT'):
   if key in os.environ:child_env[key]=os.environ[key]
  child_env.update(HUD04_FORMAL_PLAN=str(plan_path),HUD04_FORMAL_TRAIN_DEADLINE=str(budget.started+budget.seconds*.75))
  state['stage']='preflight'
  code="import torch,numpy;assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4';assert torch.cuda.device_count()==1;p=torch.cuda.get_device_properties(0);assert '4090' in p.name and [p.major,p.minor]==[8,9];print('PINNED_GPU_PREFLIGHT_OK',flush=True)"
  probe=budget.run('preflight',[request['vendor_python'],'-c',code],log=out/'preflight.log',env=child_env,cwd=root,fraction=.30)
  if probe['returncode']!=0:raise RuntimeError('Pinned GPU preflight failed')
  child_request={**request,'formal_plan_path':str(plan_path)};child_path=out/'formal-child-request.json';write_json(child_path,child_request)
  state['stage']='native_training';state['native_training_attempted']=True
  child=budget.run('native',[request['vendor_python'],'-m','student_adapter.formal_child','--request',str(child_path),'--request-sha256',sha(child_path)],
   log=out/'native.log',env=child_env,cwd=root,fraction=.88)
  write_json(out/'native-exit.json',child);state['stage']='native_validation'
  result=validate_native_result(out,request,child);state.update(training_executed=True,native_validation_passed=True,additional_updates_completed=result['additional_updates_completed'])
 except Exception as error:
  state.update(failure_type=type(error).__name__) # avoid secret-bearing exception text
 try:
  state['elapsed_seconds']=time.monotonic()-entered;write_json(out/'worker-result.json',state)
  receipt=seal_outputs(out,budget)
  receipt.update(native_validation_passed=state.get('native_validation_passed') is True,training_executed=state['training_executed'],
   _wb_agent=state['_wb_agent'],skills_evaluated=False)
  if publisher is None:raise RuntimeError('Private result publisher not configured')
  publication=publisher(Path(receipt['archive']),receipt,budget.remaining())
  if publication.get('readback_verified') is not True or publication.get('archive_sha256')!=receipt['archive_sha256']:raise ValueError('Result upload readback failed')
  budget.require();state.update(publication_verified=True,publication_receipt=publication,worker_passed=state.get('native_validation_passed') is True)
 except Exception as error:state.update(delivery_failure_type=type(error).__name__,worker_passed=False)
 state['elapsed_seconds']=time.monotonic()-entered
 # Final delivery receipt outside sealed archive avoids circular self-hashes.
 try:write_json(out.parent/(out.name+'.receipt.json'),{**state,'sealed_outputs':receipt})
 finally:budget.close()
 return state

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True,type=Path);a=p.parse_args()
 try:validate_request(json.loads(a.request.read_text()));status='request_validated_only'
 except (ValueError,KeyError,PermissionError) as error:status='blocked:'+type(error).__name__
 print(json.dumps({'status':status,'gpu_task_started':False,'operation':'validation_only'}),flush=True)

if __name__=='__main__':main()