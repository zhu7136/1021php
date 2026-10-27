"""One bounded 33-wheel isolated installation/import; never starts simulation."""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
from fetch_official_wheels import absolute_budget,load_manifest
from incremental import PIN,fetch_one
from probe_common import environment,run_json

WORKER_SECONDS=420
def fresh(path):
 path=Path(path).absolute()
 if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('symlink output')
 path.mkdir(exist_ok=False);return path
def lock_text(rows):
 if len(rows)!=33 or len({r['name'] for r in rows})!=33:raise ValueError('wrong frozen install scope')
 return ''.join(r['name']+'=='+r['version']+' --hash=sha256:'+r['sha256']+'\n' for r in rows)
def command(argv,output,deadline,limit):
 seconds=min(limit,deadline-time.monotonic())
 if seconds<2:raise TimeoutError('insufficient subprocess budget')
 process=None
 with Path(output).open('xb') as log:
  try:
   process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=environment(),start_new_session=True)
   code=process.wait(timeout=seconds)
  except BaseException:
   if process is not None and process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
   raise
 raw=Path(output).read_bytes()
 if len(raw)>4*1024*1024:raise ValueError('subprocess diagnostic exceeds limit')
 text=raw.decode('utf-8',errors='replace');events=[]
 for line in text.splitlines():
  if not line.startswith('{'):continue
  try:row=json.loads(line)
  except ValueError:continue
  if isinstance(row,dict) and str(row.get('event','')).startswith('hud04_'):events.append(row)
 return dict(returncode=code,log_bytes=len(raw),log_sha256=hashlib.sha256(raw).hexdigest(),events=events,tail=text[-2500:] if code else '')
def workflow(root,task,deadline,emit):
 manifest=load_manifest(root/'official-wheel-manifest.json',PIN);rows=manifest['files'];text=lock_text(rows)
 image=run_json("import importlib.metadata as m,importlib.util as u,json,platform;print(json.dumps(dict(python_version=platform.python_version(),torch_version=m.version('torch'),torch_origin=u.find_spec('torch').origin)),flush=True)",'/isaac-sim/python.sh',deadline,30)
 if image['python_version']!='3.11.13' or image['torch_version']!='2.7.0+cu128':raise ValueError('image interpreter/torch version differs')
 emit('hud04_isolation_image',**image)
 output=fresh(root/('isolation-'+task));wheels=fresh(output/'wheels');records=[];download_deadline=min(deadline-160,time.monotonic()+210)
 for row in rows:
  if download_deadline-time.monotonic()<2:raise TimeoutError('download total budget exhausted')
  receipt=fetch_one(row,wheels,min(download_deadline,time.monotonic()+60),emit);records.append(receipt)
  (output/'partial-receipt.json').write_text(json.dumps(dict(manifest_sha256=PIN,records=records))+'\n')
  if not receipt['passed']:raise ValueError('frozen wheel download failed')
 emit('hud04_isolation_download_complete',files_verified=len(records),verified_bytes=sum(r['received_bytes'] for r in records),manifest_sha256=PIN)
 lock=output/'requirements.lock';lock.write_text(text);target=output/'site'
 if target.exists() or target.is_symlink():raise ValueError('install target reused')
 # No index, dependency resolution, global installation, or Torch/CUDA wheel change.
 argv=['/isaac-sim/python.sh','-m','pip','--isolated','install','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(wheels),'--target',str(target),'-r',str(lock)]
 emit('hud04_isolation_install_start',lock_sha256=hashlib.sha256(text.encode()).hexdigest(),isolated_target=str(target),remaining_seconds=round(deadline-time.monotonic(),3))
 installed=command(argv,output/'install.log',deadline-65,120);emit('hud04_isolation_install_result',**installed)
 if installed['returncode']!=0:raise ValueError('isolated pip failed')
 imports=command(['/isaac-sim/python.sh',str(root/'import_check.py'),'--target',str(target),'--manifest',str(root/'official-wheel-manifest.json')],output/'imports.log',deadline-8,55)
 for event in imports.pop('events'):emit(event.pop('event'),**event)
 emit('hud04_isolation_import_process_result',**imports)
 if imports['returncode']!=0:raise ValueError('isolated import failed')
 # Independently require the expected positive child receipt, not exit0 alone.
 child=[json.loads(line) for line in (output/'imports.log').read_text().splitlines() if line.startswith('{')];final=[r for r in child if r.get('event')=='hud04_isolation_import_final']
 if len(final)!=1 or not final[0]['passed'] or not final[0]['numpy_torch_interop_passed'] or len(final[0]['isolated_distributions'])!=33:raise ValueError('import proof absent')
 for row in rows:
  path=wheels/row['filename']
  if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('wheel mutated after installation')
 return dict(files_verified=33,verified_bytes=sum(r['received_bytes'] for r in records),isolated_install_passed=True,import_passed=True,numpy_torch_interop_passed=True,image_torch_preserved=True,whole_33_files_same_worker_verified=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--task-id');parser.add_argument('--commit');args=parser.parse_args()
 if not args.execute:
  print(json.dumps(dict(execution_requested=False,network_requested=False,worker_seconds_max=WORKER_SECONDS,files_requested=33,installation_target='new isolated directory only',automatic_retries=0,simulator_started=False,ppo_updates=0)),flush=True);return 0
 if not args.task_id or not re.fullmatch('TASK_[0-9]{8}_[0-9]+',args.task_id) or not args.commit or not re.fullmatch('[0-9a-f]{40}',args.commit):raise ValueError('task/commit required')
 start=time.monotonic();deadline=start+WORKER_SECONDS;root=Path(__file__).resolve().parent
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=environment(),timeout=10,text=True).strip()
 if head!=args.commit:raise ValueError('source mismatch')
 def emit(event,**data):print(json.dumps(dict(event=event,task_id=args.task_id,source_commit=head,**data),separators=(',',':')),flush=True)
 state=dict(worker_seconds_max=WORKER_SECONDS,automatic_retries=0,simulator_started=False,ppo_updates=0,new_checkpoint=False,wandb_run_created=False,full_runtime_closure_proven=False,passed=False)
 try:
  with absolute_budget(max(.001,deadline-time.monotonic()-4)):
   state.update(workflow(root,args.task_id,deadline,emit));state['passed']=True
 except Exception as error:state.update(error_type=type(error).__name__,error_message=str(error)[:400],frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-8:]])
 state['elapsed_seconds']=round(time.monotonic()-start,3);emit('hud04_isolation_final',**state);return 0 if state['passed'] else 1
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as error:print(json.dumps(dict(event='hud04_isolation_entry_failure',error_type=type(error).__name__,passed=False)),flush=True);sys.exit(1)