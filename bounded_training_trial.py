"""Generic private single-GPU engineering trial; 1800s, no automatic retries."""
import argparse,base64,hashlib,importlib.util,json,os,signal,subprocess,sys,threading,time,zipfile
from pathlib import Path,PurePosixPath
from scoped_transport import deadline,digest,safe_unpack,transfer,validate_capability
from range_input import download

SCOPE={'max_tasks':1,'max_training_processes':1,'num_envs':10,'training_updates':100,
       'save_interval':25,'worker_seconds_max':1800,'automatic_retries':0,'fresh_student':True}

def unpack(path,destination):
 safe_unpack(path,destination)
 root=Path(destination);m=json.loads((root/'payload-manifest.json').read_text());expected={'payload-manifest.json'}
 if m['scope']!=SCOPE:raise ValueError('Trial scope mismatch')
 for r in m['files']:
  target=(root/r['path']).resolve()
  if not target.is_relative_to(root.resolve()) or r['path'] in expected:raise ValueError('Unsafe manifest')
  expected.add(r['path'])
  if target.stat().st_size!=r['bytes'] or digest(target)!=r['sha256']:raise ValueError('Input integrity')
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 if actual!=expected:raise ValueError('Unlisted input')
 return m

def emit(event,**data):print(json.dumps({'event':event,**data}),flush=True)

def main():
 started=time.monotonic();active=[None];lock=threading.Lock()
 def register(child):
  with lock:active[0]=child
 def guard():
  time.sleep(max(0,started+1800-time.monotonic()))
  with lock:
   child=active[0]
   if child is not None and child.poll() is None:
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
  os._exit(124) # failure only, never force a successful exit
 threading.Thread(target=guard,daemon=True).start()
 class Parser(argparse.ArgumentParser):
  def error(self,message):raise ValueError('Invalid private execution parameters')
 p=Parser();p.add_argument('--commit',required=True);p.add_argument('--output-cap',required=True)
 source=p.add_mutually_exclusive_group(required=True);source.add_argument('--inputs');source.add_argument('--shard-dir')
 p.add_argument('--shard-manifest-sha256')
 p.add_argument('--execution');p.add_argument('--task-id')
 p.add_argument('--resume-cap')
 a=p.parse_args();inputs=[];output=json.loads(base64.urlsafe_b64decode(a.output_cap))
 scope=SCOPE
 resume_input=None
 if a.resume_cap:
  if not a.execution or not a.shard_dir:raise ValueError('Continuation requires bound Git inputs')
  from resume_support import SCOPE as RESUME_SCOPE,CHECKPOINT_SHA256,CHECKPOINT_BYTES
  scope=RESUME_SCOPE;resume_input=json.loads(base64.urlsafe_b64decode(a.resume_cap))
  if set(resume_input)!={'bytes','sha256','cap'} or resume_input['bytes']!=CHECKPOINT_BYTES or resume_input['sha256']!=CHECKPOINT_SHA256:raise ValueError('Exact verified source checkpoint required')
  validate_capability(resume_input['cap'],'GET')
 if bool(a.shard_dir)!=bool(a.shard_manifest_sha256):raise ValueError('Local shard mode requires manifest pin')
 if bool(a.execution)!=bool(a.task_id) or (a.execution and not a.shard_dir):raise ValueError('Private execution binding requires Git shards and task identity')
 binding=None
 if a.execution:
  binding=json.loads(base64.urlsafe_b64decode(a.execution))
  if set(binding)!={'bytes','sha256','cap'} or type(binding['bytes']) is not int or not 0<binding['bytes']<=16384:raise ValueError('Small execution binding required')
  validate_capability(binding['cap'],'GET')
 if a.inputs:
  inputs=json.loads(base64.urlsafe_b64decode(a.inputs))
  if len(inputs)!=3 or [r['kind'] for r in inputs]!=['core','train_motion','eval_motion']:raise ValueError('Exactly three scoped inputs')
  for r in inputs:validate_capability(r['cap'],'GET')
 validate_capability(output,'PUT')
 root=Path(__file__).resolve().parent
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()!=a.commit:raise ValueError('Generic code pin')
 work=root/'private-trial';work.mkdir(exist_ok=False);out=work/'output';out.mkdir()
 state={'worker_passed':False,'pipeline_passed':False,'scope':scope,'code_commit':a.commit,'stage':'inputs',
        'input_pins':[{k:r[k] for k in ('kind','bytes','sha256')} for r in inputs],
        'input_mode':'local_git_shards' if a.shard_dir else 'scoped_https',
        'shard_manifest_sha256':a.shard_manifest_sha256}
 try:
  payload=work/'payload';emit('trial_stage',stage='inputs',worker_seconds_max=1800)
  if a.shard_dir:
   from local_trial_inputs import materialize
   shard_dir=Path(a.shard_dir)
   if not shard_dir.is_absolute():shard_dir=root/shard_dir
   state['input_pins']=materialize(shard_dir,work,manifest_sha256=a.shard_manifest_sha256,
      absolute_deadline=started+600,unpack_archive=unpack)
   emit('trial_local_inputs_verified',files=len(state['input_pins']))
  else:
   for index,r in enumerate(inputs):
    if index==0:path=work/'core.zip'
    else:
     name='train-motion.npz' if index==1 else 'eval-motion.npz';path=payload/'formal_data'/name
    download(r['cap'],path,size=r['bytes'],sha256=r['sha256'],absolute_deadline=started+600,
             on_progress=lambda row:emit('trial_input_progress',kind=r['kind'],**row))
    if index==0:
     with deadline(started+600):unpack(path,payload)
  execution=json.loads((payload/'execution.json').read_text())
  if resume_input is not None:
   from resume_support import make_template
   template=json.loads((root/'continuation-template.json').read_text())
   if template!=make_template(execution):raise ValueError('Continuation template differs from the unchanged base')
   execution=template
  if binding is not None:
   from execution_binding import bind_execution
   fresh=work/'execution-binding.json'
   transfer(binding['cap'],fresh,method='GET',size=binding['bytes'],sha256=binding['sha256'],absolute_deadline=started+600)
   if resume_input is None:execution=bind_execution(execution,json.loads(fresh.read_text()),a.task_id,SCOPE)
   else:
    from resume_support import bind
    execution=bind(execution,json.loads(fresh.read_text()),a.task_id)
   (payload/'execution.json').write_text(json.dumps(execution,sort_keys=True))
   state.update(task_id=a.task_id,execution_binding_sha256=binding['sha256'])
   emit('trial_execution_binding_verified',task_id=a.task_id)
  if resume_input is not None:
   transfer(resume_input['cap'],payload/'resume.pt',method='GET',size=resume_input['bytes'],sha256=resume_input['sha256'],absolute_deadline=started+600)
   overlay={'formal_worker.py':'student_adapter/formal_worker.py','formal_algo.py':'student_adapter/formal_algo.py',
            'bounded_resume.py':'student_adapter/bounded_resume.py','engineering_worker.py':'engineering_worker.py'}
   for source_name,target_name in overlay.items():
    source=root/'resume-overlay'/source_name;target=payload/target_name
    if source.is_symlink() or not source.is_file() or target.is_symlink():raise ValueError('Invalid pinned resume overlay')
    target.write_bytes(source.read_bytes())
   state['resume_checkpoint_sha256']=resume_input['sha256']
   emit('trial_resume_inputs_verified',source_completed_updates=100,max_additional_updates=1000)
  if (execution.get('plan',{}).get('execution_approved') is not True or
      execution.get('approval',{}).get('confirmed_by_user') is not True or
      not isinstance(execution.get('task_id'),str) or not execution['task_id'].startswith('TASK_')):
   raise ValueError('Fresh task-specific training approval required')
  state['stage']='training_worker';sys.path.insert(0,str(payload))
  spec=importlib.util.spec_from_file_location('private_engineering_worker',payload/'engineering_worker.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  result=module.run(payload,out,started=started,on_spawn=register)
  state['worker_passed']=result.get('native_validation_passed') is True
  state['stage']='complete' if state['worker_passed'] else 'training_worker_failed'
 except Exception as exc:
  state['error_type']=type(exc).__name__;emit('trial_failure',stage=state['stage'],error_type=type(exc).__name__)
 state['elapsed_seconds_before_packaging']=time.monotonic()-started
 (out/'bootstrap-result.json').write_text(json.dumps(state,indent=2))
 with deadline(started+1710):
  files=[]
  for path in sorted(out.rglob('*')):
   if path.is_symlink():raise ValueError('Result symlink')
   if path.is_file():files.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':digest(path)})
  if sum(r['bytes'] for r in files)>500*1024**2:raise ValueError('Result bounds')
  (out/'result-manifest.json').write_text(json.dumps({'scope':scope,'files':files},sort_keys=True))
  archive=work/'trial-result.zip'
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for path in sorted(out.rglob('*')):
    if path.is_file():z.write(path,path.relative_to(out))
 receipt=transfer(output,archive,method='PUT',size=archive.stat().st_size,sha256=digest(archive),absolute_deadline=started+1790)
 emit('trial_result_uploaded',**receipt,worker_passed=state['worker_passed'])
 return 0 if state['worker_passed'] else 1

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:emit('trial_bootstrap_failure',error_type=type(exc).__name__);sys.exit(1)