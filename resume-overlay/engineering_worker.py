"""Private one-shot fresh 100-update engineering trial; no scheduler or keys."""
import hashlib,json,os,sys,time
from pathlib import Path
from student_adapter.formal_worker import validate_request,write_json,verify_runtime_inputs,validate_native_result
from student_adapter.lifecycle_probe_parent import capture_process
from student_adapter.worker_budget import isolated_child_env
from student_adapter.runtime_imports import runtime_pythonpath

SCOPE={'max_tasks':1,'max_training_processes':1,'num_envs':10,'training_updates':100,
       'save_interval':25,'worker_seconds_max':1800,'automatic_retries':0,'fresh_student':True}

def run(root,out,*,started,on_spawn):
 root,out=Path(root),Path(out)
 source=json.loads((root/'execution.json').read_text())
 from student_adapter.bounded_resume import SCOPE as RESUME_SCOPE,CHECKPOINT_SHA256
 continuing=source['scope']==RESUME_SCOPE
 if source['scope'] not in (SCOPE,RESUME_SCOPE) or not 0<=time.monotonic()-started<600:raise ValueError('Original entry clock/scope required')
 p=source['plan'];approval=source['approval']
 state={'native_validation_passed':False,'learning_evidence_passed':False,'training_updates_observed':0,
        'stage':'install','automatic_retries':0,'task_id':source['task_id'],'_wb_agent':p['_wb_agent'],'skills_evaluated':False}
 native=None
 vendor='/isaac-sim/python.sh';bundle=root/'bundle';site=root/'dependency-site'
 py=os.pathsep.join(map(str,runtime_pythonpath(root)))
 env=isolated_child_env(os.environ,pythonpath=py,output=out,asset_root=bundle/'asset',train_deadline=started+3240,task_id=source['task_id'])
 for key in ('ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT'):
  if key in os.environ:env[key]=os.environ[key]
 def spawn(command,log,cutoff,on_tick=None):
  if time.monotonic()>=started+cutoff:raise TimeoutError('Absolute trial stage deadline')
  return capture_process(command,log=out/log,deadline=started+cutoff,env=env,cwd=root,on_spawn=on_spawn,**({'on_tick':on_tick} if on_tick is not None else {}))
 try:
  install=[vendor,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes',
           '--find-links',str(root/'deps/wheels'),'--target',str(site),'-r',str(root/'deps/overlay.lock')]
  r=spawn(install,'install.log',780);write_json(out/'install-exit.json',r)
  if r['raw_returncode']!=0 or r['timed_out']:raise RuntimeError('Pinned dependency install failed')
  state['stage']='runtime_manifest'
  hashes={}
  for path in sorted(root.rglob('*')):
   if path.is_symlink():raise ValueError('Runtime symlink')
   if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
    if time.monotonic()>=started+900:raise TimeoutError('Runtime hashing deadline')
    h=hashlib.sha256()
    with path.open('rb') as f:
     for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    hashes[str(path.relative_to(root))]=h.hexdigest()
  manifest=root/'formal-runtime-manifest.json';write_json(manifest,{'files':hashes})
  request={'schema':'hud04-formal-worker-request-v1','plan':p,'automatic_retries':0,'task_id':source['task_id'],
           'runtime_root':str(root),'bundle':str(bundle),'data_root':str(root/'formal_data'),'teacher_root':str(root/'teachers'),
           'output':str(out),'vendor_python':vendor,'runtime_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'resume':None,'approval':approval,'engineering_validation':source['engineering_validation']}
  if continuing:
   request.update(resume=str(root/'resume.pt'),resume_sha256=CHECKPOINT_SHA256,continuation_validation=source['continuation_validation'])
  validate_request(request);verify_runtime_inputs(request)
  plan_path=out/'formal-plan.json';write_json(plan_path,p)
  env.update(HUD04_FORMAL_PLAN=str(plan_path),HUD04_FORMAL_TRAIN_DEADLINE=str(started+3240))
  state['stage']='gpu_preflight'
  code="from student_adapter.runtime_imports import verify_overlay_resolution;print(verify_overlay_resolution("+repr(str(bundle))+") ,flush=True);import torch,numpy;assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4';assert torch.cuda.device_count()==1;p=torch.cuda.get_device_properties(0);assert '4090' in p.name and [p.major,p.minor]==[8,9] and 23<=p.total_memory/2**30<=25;print(p.name,p.total_memory,flush=True)"
  r=spawn([vendor,'-c',code],'gpu-preflight.log',960);write_json(out/'preflight-exit.json',r)
  if r['raw_returncode']!=0 or r['timed_out']:raise RuntimeError('Pinned GPU preflight failed')
  child_request={**request,'formal_plan_path':str(plan_path)};path=out/'formal-child-request.json';write_json(path,child_request)
  state['stage']='native_training';print(json.dumps({'event':'trial_stage','stage':'native_training','updates':p['additional_updates'],'num_envs':10,'continuing':continuing}),flush=True)
  from student_adapter.training_telemetry import ProgressMirror
  mirror=ProgressMirror(out,source['task_id'],p['_wb_agent'])
  native=spawn([vendor,'-m','student_adapter.formal_child','--request',str(path),'--request-sha256',hashlib.sha256(path.read_bytes()).hexdigest()],'native.log',3360,on_tick=mirror)
  state['metrics_relay_completed_updates']=mirror.last
  state['metrics_relay_error_type']=native.get('telemetry_error_type')
  write_json(out/'native-exit.json',native)
  progress=list(out.rglob('formal_progress.jsonl'))
  if len(progress)==1:
   rows=[json.loads(line) for line in progress[0].read_text().splitlines() if line.strip()]
   state['training_updates_observed']=len(rows)
  result_path=out/'formal-native-result.json'
  if result_path.is_file():
   result=json.loads(result_path.read_text());checkpoint=(out/result['checkpoint']).resolve()
   if checkpoint.is_relative_to(out.resolve()) and checkpoint.is_file():
    h=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    added=result['additional_updates_completed'];start=1100 if continuing else 0
    state['learning_evidence_passed']=1<=added<=p['additional_updates'] and result['completed_updates']==start+added and result['teachers_frozen_verified'] is True and result['checkpoint_model_readback_verified'] is True and result['_wb_agent']==p['_wb_agent'] and h==result['checkpoint_sha256']
    if continuing:
     restored=json.loads((out/'formal-resume-verification.json').read_text())
     state['resume_verified']=restored['start_completed_updates']==1100 and restored['checkpoint_sha256']==CHECKPOINT_SHA256 and restored['optimizer_restore_and_readback'] is True
     state['learning_evidence_passed']=state['learning_evidence_passed'] and state['resume_verified']
  if continuing and state['learning_evidence_passed']:
   from student_adapter.bounded_resume import validate_outputs
   state['learning_evidence_passed']=False
   validate_outputs(out,request)
   state['learning_evidence_passed']=True
   # Learning and durable artifacts remain distinct from the true process exit.
   state['native_lifecycle_exit_passed']=native['raw_returncode']==0 and not native['timed_out']
   if state['native_lifecycle_exit_passed']:
    lifecycle=json.loads((out/'formal-lifecycle.json').read_text())
    state['native_lifecycle_exit_passed']=all(lifecycle.get(k) is True for k in ['train_returned','close_returned','child_returned','fast_shutdown_disabled_verified']) and lifecycle['_wb_agent']==p['_wb_agent']
   state['native_validation_passed']=state['native_lifecycle_exit_passed'];state['stage']='complete' if state['native_lifecycle_exit_passed'] else 'learning_complete_shutdown_failed'
   write_json(out/'engineering-worker-result.json',dict(state,native_raw_returncode=native['raw_returncode'],elapsed_seconds_before_packaging=time.monotonic()-started))
   return state
  state['stage']='native_validation'
  validate_native_result(out,request,{'returncode':native['raw_returncode'],'timed_out':native['timed_out']})
  state['native_validation_passed']=True;state['stage']='complete'
 except Exception as exc:
  state['error_type']=type(exc).__name__
 state['native_raw_returncode']=None if native is None else native['raw_returncode']
 state['elapsed_seconds_before_packaging']=time.monotonic()-started
 write_json(out/'engineering-worker-result.json',state)
 return state