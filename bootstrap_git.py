"""Bounded zero-PPO four-skill diagnostic; unchanged approved inputs."""
import argparse,base64,json,os,re,signal,subprocess,sys,threading,time,zipfile,shutil
from pathlib import Path
from bootstrap_base import safe_extract,clean_env,SafeParser
from scoped_transport import transfer,digest,validate_capability,deadline
from git_parts_reassemble import assemble
INPUT_BYTES=116980683
INPUT_SHA256='9eb6f16d5f5630f263d67d6d0e084724d7a2d3ac4836ce7b448c658c6d435e4f'
VENDOR='/isaac-sim/python.sh';ACTIVE=set()
ALLOWED_OVERLAY={'bundle/terrain/validation-four.obj','bundle/reference/validation-four.npz','validation-code/config.json','validation-code/cases.json','validation-code/validation_child.py'}
CHILD_SHA256='bb09b105b2faed0a04db473d46038862fe5d80d2aa2dd17ab7ce07195652677d'
def install_child_override(root,payload):
 p=root/'validation_child_proxy_fix.py'
 if p.is_symlink() or not p.is_file() or digest(p)!=CHILD_SHA256:raise ValueError('Code-only child override integrity mismatch')
 shutil.copyfile(p,payload/'validation-code/validation_child.py')
 if digest(payload/'validation-code/validation_child.py')!=CHILD_SHA256:raise ValueError('Installed child hash mismatch')
 return CHILD_SHA256
def local_overlay(root,expected_bytes,expected_sha256):
 p=root/'git-input/hud04-current-four-skills/overlay.zip'
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=expected_bytes or digest(p)!=expected_sha256:raise ValueError('Git overlay integrity mismatch')
 return p
def install_overlay(archive,overlay,payload):
 m=safe_extract(archive,overlay)
 if {r['path'] for r in m['files']}!=ALLOWED_OVERLAY:raise ValueError('Unexpected overlay paths')
 if m['scope']['training_update_cap']!=0 or m['scope']['worker_seconds_max']!=600 or m['scope']['automatic_retries']!=0:raise ValueError('Unexpected overlay scope')
 cases=json.loads((overlay/'validation-code/cases.json').read_text())
 if {c['dataset'] for c in cases}!={'motions-locomotion','motions-high-step','motions-low-climb-76','motions-high-climb-76'} or len(cases)!=4:raise ValueError('Wrong four skills')
 expected=sum(len(c['probe_frames']) for c in cases)
 if expected!=12:raise ValueError('Unexpected probe count')
 for row in m['files']:shutil.copyfile(overlay/row['path'],payload/row['path'])
 return expected
def stage(cmd,log,env,end):
 if time.monotonic()>=end:raise TimeoutError('Stage budget exhausted')
 with Path(log).open('xb') as stream:
  p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);ACTIVE.add(p.pid)
  try:
   while p.poll() is None:
    if time.monotonic()>=end:raise TimeoutError('Stage deadline')
    time.sleep(.1)
   return p.returncode
  finally:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait(timeout=2);ACTIVE.discard(p.pid)
def watchdog(start):
 time.sleep(max(0,start+598-time.monotonic()))
 for pid in list(ACTIVE):
  try:os.killpg(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 os._exit(124)
def parse(argv=None):
 p=SafeParser(description=__doc__);p.add_argument('--execute',action='store_true')
 for n in ['task-id','commit','output-cap','overlay-sha256','thread-id','turn-id']:p.add_argument('--'+n)
 p.add_argument('--overlay-bytes',type=int);a=p.parse_args(argv)
 if a.execute:
  if not all(vars(a).values()):raise ValueError('Missing execution bounds')
  if not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id) or not re.fullmatch('[0-9a-f]{40}',a.commit) or not re.fullmatch('[0-9a-f]{64}',a.overlay_sha256):raise ValueError('Invalid binding')
  if not 0<a.overlay_bytes<30*1024**2 or len(a.output_cap)>16384:raise ValueError('Oversized capability/overlay')
 return a
def main(argv=None):
 start=time.monotonic();a=parse(argv)
 if not a.execute:
  print(json.dumps({'execution_requested':False,'worker_seconds_max':600,'ppo_updates':0,'input_transport':'git-base+approved-git-overlay','clone_overhead_bounded':False}));return 0
 threading.Thread(target=watchdog,args=(start,),daemon=True).start()
 co=json.loads(base64.urlsafe_b64decode(a.output_cap));validate_capability(co,'PUT')
 env=clean_env();root=Path(__file__).resolve().parent
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=env,text=True,timeout=5).strip()!=a.commit:raise ValueError('Source pin mismatch')
 work=root/('private-'+a.task_id);work.mkdir(exist_ok=False)
 state={'task_id':a.task_id,'bootstrap_commit':a.commit,'worker_seconds_max':600,'automatic_retries':0,'ppo_updates':0,'worker_passed':False,'stage':'git_input','teacher_converged':False,'input_transport':'git-base+approved-git-overlay','input_hash_verified':False,'overlay_hash_verified':False,'native_simulator_started':False,'clone_overhead_bounded':False}
 try:
  print(json.dumps({'event':'stage','stage':'git_input','task_id':a.task_id}),flush=True)
  with deadline(start+120):
   state['git_input']=assemble(root/'git-input/hud04-validation-v3',work/'input.zip',expected_bytes=INPUT_BYTES,expected_sha256=INPUT_SHA256,absolute_deadline=start+120)
   m=safe_extract(work/'input.zip',work/'payload');payload=work/'payload'
   if m['scope']['training_update_cap']!=0 or m['scope']['worker_seconds_max']!=600:raise ValueError('Unexpected base scope')
   state['input_hash_verified']=True;state['stage']='git_overlay'
   print(json.dumps({'event':'stage','stage':'git_overlay'}),flush=True)
   archive=local_overlay(root,a.overlay_bytes,a.overlay_sha256)
   expected=install_overlay(archive,work/'overlay',payload);state['overlay_hash_verified']=True
   state['child_sha256']=install_child_override(root,payload)
  state['stage']='install';print(json.dumps({'event':'stage','stage':'install'}),flush=True);site=work/'dependency-site'
  if not Path(VENDOR).is_file():raise ValueError('Vendor Python missing')
  if stage([VENDOR,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')],work/'install.log',env,start+210):raise ValueError('Offline installation failed')
  env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','');env['PHP_FLUX_TASK_ID']=a.task_id
  state['stage']='native_validation';print(json.dumps({'event':'stage','stage':'native_validation'}),flush=True);out=work/'native-output';out.mkdir();state['native_simulator_start_attempted']=True
  rc=stage([VENDOR,str(payload/'validation-code/validation_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--thread-id',a.thread_id,'--turn-id',a.turn_id],out/'native.log',env,start+515)
  result=json.loads((out/'validation-report.json').read_text())
  if result['task_id']!=a.task_id or result['ppo_updates']!=0 or result['learner_constructed'] is not False:raise ValueError('Native result scope/binding mismatch')
  state.update(native_exit_code=rc,native_simulator_started=result.get('native_environment_loaded',False),native_validation_completed=result.get('native_validation_completed',False),worker_passed=bool(rc==0 and result.get('load_reset_short_physics_passed') and result.get('simulator_close_returned') and result.get('probe_count')==expected and len(result.get('cases',[]))==4 and result.get('fast_shutdown_disabled_verified') is True and result.get('fast_shutdown_at_close') is False),stage='complete')
 except Exception as e:
  state['error_type']=type(e).__name__;print(json.dumps({'event':'stage_failure','stage':state['stage'],'error_type':type(e).__name__}),flush=True)
  partial=work/'native-output/validation-report.json'
  if partial.is_file():
   try:
    result=json.loads(partial.read_text())
    if result.get('task_id')==a.task_id:state.update(native_simulator_started=result.get('native_environment_loaded',False),native_probe_count=result.get('probe_count',0),native_child_error_type=result.get('error_type'),simulator_close_returned=result.get('simulator_close_returned',False))
   except (ValueError,OSError):pass
 state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3);(work/'bootstrap-status.json').write_text(json.dumps(state,indent=2));files=[]
 for f in sorted(work.rglob('*')):
  if f.is_symlink() or not f.is_file() or f.relative_to(work).parts[0] in {'payload','dependency-site','overlay'} or f.suffix not in {'.json','.jsonl','.log'}:continue
  files.append({'path':str(f.relative_to(work)),'bytes':f.stat().st_size,'sha256':digest(f)})
 if sum(f['bytes'] for f in files)>=30*1024**2:raise ValueError('Result size limit')
 result=work/'flux-result.zip'
 with zipfile.ZipFile(result,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  z.writestr('result-manifest.json',json.dumps({'task_id':a.task_id,'source_commit':a.commit,'files':files,'ppo_updates':0,'worker_seconds_max':600,'automatic_retries':0,'input_transport':'git-base+approved-git-overlay','input_sha256':INPUT_SHA256,'overlay_sha256':a.overlay_sha256,'child_sha256':CHILD_SHA256}))
  for f in files:z.write(work/f['path'],f['path'])
 receipt=transfer(co,result,method='PUT',size=result.stat().st_size,sha256=digest(result),absolute_deadline=start+590)
 print(json.dumps({'event':'result_uploaded','task_id':a.task_id,'bytes':receipt['bytes'],'sha256':receipt['sha256'],'worker_passed':state['worker_passed'],'elapsed_seconds':round(time.monotonic()-start,3)}),flush=True)
 return 0 if state['worker_passed'] else 1
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:print(json.dumps({'event':'bootstrap_failure','error_type':type(e).__name__}),flush=True);sys.exit(1)