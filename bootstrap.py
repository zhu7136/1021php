"""One full-skill fresh teacher, 4096env, 1800s total worker, zero retries."""
import argparse,base64,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
from bootstrap_base import safe_extract,clean_env
from scoped_transport import transfer,digest,validate_capability
from qq_input import validate_capability as qq_validate
import transfer_diagnostics
transfer_diagnostics.validate_capability=qq_validate
from transfer_diagnostics import download as base_download
from range_input import download as private_download
from formal_bootstrap import stage,pack,watchdog,emit
BASE_BYTES=123410856
BASE_SHA='490d7d1b8b927f42303816c1a40f440c05ef5393aec950724ab897fad45c643f'
ALLOWED={'bundle/reference/full-reference.npz','bundle/terrain/full-terrain.obj','teacher-code/config.json','teacher-code/cases.json','teacher-code/teacher_child.py','teacher-code/teacher_algo.py','teacher-code/close_audit.py'}
def parse(argv=None):
 from bootstrap_base import SafeParser
 p=SafeParser();p.add_argument('--execute',action='store_true')
 for n in ['task-id','commit','base-cap','input-cap','output-cap','input-sha256','thread-id','turn-id','skill']:p.add_argument('--'+n)
 p.add_argument('--input-bytes',type=int);a=p.parse_args(argv)
 if a.execute:
  import re
  if not all(vars(a).values()) or a.skill not in ['locomotion','high-step'] or not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id) or not re.fullmatch('[0-9a-f]{40}',a.commit) or not re.fullmatch('[0-9a-f]{64}',a.input_sha256) or not 0<a.input_bytes<512*1024**2:raise ValueError('Missing or invalid execution binding')
 return a
def main(argv=None):
 start=time.monotonic();a=parse(argv)
 if not a.execute:emit('execution_not_requested',worker_seconds_max=1800,automatic_retries=0);return 0
 caps={n:json.loads(base64.urlsafe_b64decode(getattr(a,n+'_cap'))) for n in ['base','input','output']}
 qq_validate(caps['base'],'GET');validate_capability(caps['input'],'GET');validate_capability(caps['output'],'PUT')
 root=Path(__file__).resolve().parent;env=clean_env()
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=env,text=True,timeout=10).strip()!=a.commit:raise ValueError('Source commit mismatch')
 threading.Thread(target=watchdog,args=(start,),daemon=True).start()
 work=root/('private-'+a.task_id);work.mkdir(exist_ok=False)
 state={'task_id':a.task_id,'bootstrap_commit':a.commit,'worker_seconds_max':1800,'automatic_retries':0,'requested_num_envs':4096,'skill':a.skill,'fresh_training':True,'teacher_converged':False,'worker_passed':False,'stage':'private_full_input','input_verified':False,'native_training_started':False}
 try:
  emit('stage',stage=state['stage'],task_id=a.task_id)
  state['private_input_transfer']=private_download(caps['input'],work/'full-input.zip',size=a.input_bytes,sha256=a.input_sha256,absolute_deadline=start+240)
  overlay=work/'overlay';manifest=safe_extract(work/'full-input.zip',overlay);scope=manifest['scope']
  expected=100 if a.skill=='locomotion' else 60
  if {r['path'] for r in manifest['files']}!=ALLOWED or scope['skill']!=a.skill or scope['clip_count']!=expected or scope['num_envs']!=4096 or scope['worker_seconds_max']!=1800 or scope['automatic_retries']!=0 or not scope['fresh_training']:raise ValueError('Full input scope mismatch')
  cases=json.loads((overlay/'teacher-code/cases.json').read_text())
  if len(cases)!=expected or sum(c['frames'] for c in cases)!=scope['frame_count']:raise ValueError('Full dataset coverage mismatch')
  state.update(clip_count=expected,frame_count=scope['frame_count'],private_input_verified=True,stage='base_runtime')
  emit('stage',stage=state['stage'],private_input_verified=True,clip_count=expected,frame_count=scope['frame_count'])
  state['base_transfer']={}
  base_download(caps['base'],work/'input.zip',size=BASE_BYTES,sha256=BASE_SHA,absolute_deadline=min(start+360,time.monotonic()+120),progress=state['base_transfer'])
  payload=work/'payload';base=safe_extract(work/'input.zip',payload)
  if base['scope']['num_envs']!=128 or base['scope']['worker_seconds_max']!=600:raise ValueError('Base source mismatch')
  # Only named full-reference, full-terrain and new teacher-code paths added;
  # all inherited source assets and original reference stay intact.
  for r in manifest['files']:
   target=payload/r['path'];target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():raise ValueError('Refuse to overwrite an inherited input')
   shutil.copyfile(overlay/r['path'],target)
  state.update(input_verified=True,stage='install');emit('stage',stage='install')
  vendor='/isaac-sim/python.sh';site=work/'dependency-site'
  if not Path(vendor).is_file():raise ValueError('Vendor interpreter missing')
  rc=stage([vendor,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')],work/'install.log',env,start+490)
  if rc:raise ValueError('Offline install failed')
  env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','');env['PHP_FLUX_TASK_ID']=a.task_id
  state['stage']='gpu_preflight';emit('stage',stage='gpu_preflight')
  probe="import torch,numpy;p=torch.cuda.get_device_properties(0);print(p.name,p.total_memory,flush=True);assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4' and torch.cuda.device_count()==1 and '4090' in p.name and [p.major,p.minor]==[8,9] and 23<=p.total_memory/2**30<=25"
  if stage([vendor,'-c',probe],work/'gpu-preflight.log',env,start+540):raise ValueError('GPU contract failed')
  if time.monotonic()>start+650:raise TimeoutError('Insufficient training budget')
  out=work/'native-output';out.mkdir();state.update(stage='native_training',native_training_started=True)
  emit('stage',stage='native_training',requested_num_envs=4096,clip_count=expected)
  cmd=[vendor,str(payload/'teacher-code/teacher_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--thread-id',a.thread_id,'--turn-id',a.turn_id]
  rc=stage(cmd,out/'native.log',env,start+1710,stop_file=out/'STOP_AFTER_UPDATE',soft=start+1650);state['native_exit_code']=rc
  state['stage']='result_validation'
  learn=json.loads((out/'learning_state.json').read_text());life=json.loads((out/'lifecycle.json').read_text())
  required=['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite']
  state.update(completed_updates=learn['completed_updates'],checkpoint_sha256=learn['checkpoint_sha256'],close_returned=life.get('close_returned',False),train_returned=life.get('train_returned',False))
  if rc or not 1<=learn['completed_updates']<=100000 or not all(learn[k] for k in required) or life.get('task_id')!=a.task_id or not all(life.get(k) for k in ['close_returned','train_returned','child_returned','fast_shutdown_disabled_verified']):raise ValueError('Learning or normal-return evidence incomplete')
  state.update(worker_passed=True,stage='complete')
 except Exception as e:
  state['error_type']=type(e).__name__;emit('stage_failure',stage=state['stage'],error_type=type(e).__name__)
 state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3)
 result=pack(work,state)
 receipt=transfer(caps['output'],result,method='PUT',size=result.stat().st_size,sha256=digest(result),absolute_deadline=start+1790)
 emit('result_uploaded',task_id=a.task_id,bytes=receipt['bytes'],sha256=receipt['sha256'],worker_passed=state['worker_passed'],elapsed_seconds=round(time.monotonic()-start,3))
 return 0 if state['worker_passed'] else 1
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:emit('bootstrap_failure',error_type=type(e).__name__);sys.exit(1)