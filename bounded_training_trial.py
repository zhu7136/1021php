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
 p=Parser();p.add_argument('--commit',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output-cap',required=True)
 a=p.parse_args();inputs=json.loads(base64.urlsafe_b64decode(a.inputs));output=json.loads(base64.urlsafe_b64decode(a.output_cap))
 if len(inputs)!=3 or [r['kind'] for r in inputs]!=['core','train_motion','eval_motion']:raise ValueError('Exactly three scoped inputs')
 for r in inputs:validate_capability(r['cap'],'GET')
 validate_capability(output,'PUT')
 root=Path(__file__).resolve().parent
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()!=a.commit:raise ValueError('Generic code pin')
 work=root/'private-trial';work.mkdir(exist_ok=False);out=work/'output';out.mkdir()
 state={'worker_passed':False,'pipeline_passed':False,'scope':SCOPE,'code_commit':a.commit,'stage':'inputs',
        'input_pins':[{k:r[k] for k in ('kind','bytes','sha256')} for r in inputs]}
 try:
  payload=work/'payload';emit('trial_stage',stage='inputs',worker_seconds_max=1800)
  for index,r in enumerate(inputs):
   if index==0:path=work/'core.zip'
   else:
    name='train-motion.npz' if index==1 else 'eval-motion.npz';path=payload/'formal_data'/name
   download(r['cap'],path,size=r['bytes'],sha256=r['sha256'],absolute_deadline=started+600,
            on_progress=lambda row:emit('trial_input_progress',kind=r['kind'],**row))
   if index==0:
    with deadline(started+600):unpack(path,payload)
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
  (out/'result-manifest.json').write_text(json.dumps({'scope':SCOPE,'files':files},sort_keys=True))
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