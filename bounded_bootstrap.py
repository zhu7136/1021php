"""Generic hash-pinned private workload launcher. No model/data/account keys.

One worker, 600 seconds from entry, zero retries. Signed single-object tickets
are provided only in private task parameters, never committed to this file.
"""
import argparse,base64,hashlib,json,os,signal,stat,subprocess,sys,threading,time,zipfile
from pathlib import Path,PurePosixPath
from scoped_transport import deadline,digest,transfer,validate_capability
from range_input import download

def unpack(archive,destination):
 destination=Path(destination);destination.mkdir(exist_ok=False)
 with zipfile.ZipFile(archive) as z:
  rows=z.infolist();names=set()
  if len(rows)>1600 or sum(r.file_size for r in rows)>512*1024**2:raise ValueError('Archive bounds')
  for r in rows:
   p=PurePosixPath(r.filename);mode=r.external_attr>>16
   if not r.filename or p.is_absolute() or '..' in p.parts or '\\' in r.filename or r.filename in names or r.is_dir() or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0,stat.S_IFREG):raise ValueError('Archive path')
   names.add(r.filename);target=destination.joinpath(*p.parts);target.parent.mkdir(parents=True,exist_ok=True)
   with z.open(r) as a,target.open('xb') as b:
    import shutil
    shutil.copyfileobj(a,b,1024**2)
 manifest=json.loads((destination/'payload-manifest.json').read_text());expected={'payload-manifest.json'}
 for row in manifest['files']:
  p=PurePosixPath(row['path'])
  if p.is_absolute() or '..' in p.parts or row['path'] in expected:raise ValueError('Manifest path')
  expected.add(row['path']);f=destination/row['path']
  if f.stat().st_size!=row['bytes'] or digest(f)!=row['sha256']:raise ValueError('Manifest integrity')
 if names!=expected:raise ValueError('Unlisted payload')
 return manifest

def emit(event,**data):print(json.dumps({'event':event,**data}),flush=True)

def main():
 started=time.monotonic();child=[None]
 def guard():
  time.sleep(max(0,started+600-time.monotonic()))
  if child[0] is not None:
   try:os.killpg(child[0].pid,signal.SIGKILL)
   except ProcessLookupError:pass
  os._exit(124)
 threading.Thread(target=guard,daemon=True).start()
 class Parser(argparse.ArgumentParser):
  def error(self,message):raise ValueError('Invalid private execution parameters')
 p=Parser()
 for key in ('commit','input-cap','output-cap','input-sha256'):p.add_argument('--'+key,required=True)
 p.add_argument('--input-bytes',type=int,required=True)
 a=p.parse_args();caps={k:json.loads(base64.urlsafe_b64decode(getattr(a,k+'_cap'))) for k in ('input','output')}
 for key,method in [('input','GET'),('output','PUT')]:validate_capability(caps[key],method)
 root=Path(__file__).resolve().parent
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,timeout=5,text=True).strip()!=a.commit:raise ValueError('Code pin mismatch')
 work=root/'private-worker';work.mkdir(exist_ok=False);out=work/'output';out.mkdir()
 state={'worker_passed':False,'stage':'input','worker_seconds_max':600,'automatic_retries':0,'code_commit':a.commit,'input_sha256':a.input_sha256}
 try:
  emit('worker_stage',stage='input',worker_seconds_max=600)
  progress={}
  download(caps['input'],work/'input.zip',size=a.input_bytes,sha256=a.input_sha256,absolute_deadline=started+150,progress=progress,
   on_progress=lambda x:emit('input_progress',**x))
  with deadline(started+150):manifest=unpack(work/'input.zip',work/'payload')
  if manifest['scope']!={'max_training_iterations':5,'worker_seconds_max':600,'automatic_retries':0,'fresh_student':True,'trial_count':1}:raise ValueError('Scope mismatch')
  env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','LD_LIBRARY_PATH','DISPLAY','LANG','LC_ALL','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT','ACCEPT_EULA','OMNI_KIT_ACCEPT_EULA'}}
  env.update(PYTHONUNBUFFERED='1',WANDB_MODE='offline',WORKER_STARTED=str(started))
  state['stage']='private_worker';emit('worker_stage',stage='private_worker')
  with (out/'worker.log').open('xb') as log:
   child[0]=subprocess.Popen([sys.executable,str(work/'payload/worker_main.py'),'--output',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=child[0].wait(timeout=max(.01,started+540-time.monotonic()))
   except subprocess.TimeoutExpired:
    os.killpg(child[0].pid,signal.SIGKILL);child[0].wait(timeout=3);raise TimeoutError('Private worker deadline')
   finally:child[0]=None
  state['private_worker_exit_code']=rc
  result=json.loads((out/'worker-result.json').read_text())
  state['worker_passed']=rc==0 and result.get('worker_passed') is True
  state['stage']='complete' if state['worker_passed'] else 'private_worker_failed'
 except Exception as e:
  state['error_type']=type(e).__name__;emit('worker_failure',stage=state['stage'],error_type=type(e).__name__)
 state['elapsed_seconds']=round(time.monotonic()-started,3);(out/'bootstrap-result.json').write_text(json.dumps(state,indent=2))
 with deadline(started+550):
  result_zip=work/'result.zip'
  with zipfile.ZipFile(result_zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   total=0
   for f in sorted(out.rglob('*')):
    if f.is_symlink():raise ValueError('Result symlink')
    if not f.is_file():continue
    total+=f.stat().st_size
    if total>256*1024**2:raise ValueError('Result bounds')
    z.write(f,f.relative_to(out))
 receipt=transfer(caps['output'],result_zip,method='PUT',size=result_zip.stat().st_size,sha256=digest(result_zip),absolute_deadline=started+595)
 emit('result_uploaded',**receipt,worker_passed=state['worker_passed'])
 return 0 if state['worker_passed'] else 1

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:emit('bootstrap_failure',error_type=type(e).__name__);sys.exit(1)