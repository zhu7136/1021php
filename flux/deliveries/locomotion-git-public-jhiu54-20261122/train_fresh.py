"""Resume the all29 G1-matched locomotion teacher, bounded worker, zero retries.

Default: CPU-only package preflight. --execute: one remote vendor-runtime child.
No account credential or signed capability is accepted in arguments or Git.
Final selected results are sealed and mirrored to the task's private log as a
bounded, checksummed archive. Retrieval must be verified before claiming saved
remote results; platform preview alone may truncate the archive.
"""
import argparse, base64, hashlib, json, math, os, re, signal, subprocess, sys, time, zipfile
from pathlib import Path, PurePosixPath
from local_shards import reassemble

ROOT=Path(__file__).resolve().parent
PIN='5c4fa406c8c1381cd1dea9cec433b403a998cc2bfb303d0ec69248ccd4afdb4f'
LIMIT=1800
UPDATES=2000
SOURCE_SHA='b42fabd4343ca0c9b8ac21d796c99a290fcb9a096433933ebbb48586acae73af'
CHECKPOINT_PIN='27c33904b270f26c7be20c674377fda7f8f0010a643892a5b738f43f86e34162'
RESUME_PATH='/workspace/resume-input/model_smoke_final.pt'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()

def emit(name,**d):print(json.dumps(dict(event=name,**d),allow_nan=False),flush=True)

def check_rows(root,rows):
 for row in rows:
  p=root/row['path']
  if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file():raise ValueError('nonregular_package_member')
  if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('package_hash_mismatch')

def extract(zpath,target):
 with zipfile.ZipFile(zpath) as z:
  seen=set()
  for i in z.infolist():
   p=PurePosixPath(i.filename)
   if p.is_absolute() or '..' in p.parts or '\\' in i.filename or i.filename in seen:raise ValueError('unsafe_zip_member')
   if ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('zip_symlink')
   if p.suffix.lower() in {'.pt','.pth','.onnx','.ckpt','.safetensors'}:raise ValueError('historical_weight_forbidden')
   seen.add(i.filename)
  z.extractall(target)

def validate_config(c):
 t=c['training']; a=c['algo']['config']
 assert t['num_envs']==4096 and t['seed']==42 and t['headless'] and not t['multigpu']
 assert t['checkpoint']==RESUME_PATH and a['load_optimizer'] is True
 assert not any(t.get(k) for k in ['teacher_checkpoint','registry_name','wandb_path'])
 expected=json.loads((ROOT/'PLAN.json').read_text())['scope']['g1_velocity_limits_rad_s']
 assert len(expected)==29
 assert all(v==expected.get(n,13.) for n,v in zip(c['robot']['dof_names'],c['robot']['dof_vel_limit_list']))
 assert a['num_learning_iterations']==UPDATES and a['num_steps_per_env']==24
 assert len(c['robot']['dof_names'])==31 and c['robot']['actions_dim']==29
 assert 'locomotion' in t['name']
 assert c['randomization']['setup_terms']['setup_dof_pos_bias']['params']['enabled'] is False

def prepare(work,plan):
 archives=work/'archives'
 input_root=ROOT/'inputs'
 if not any(input_root.glob('*.bin')):input_root=ROOT.parent/'locomotion-g1all-fresh-20261122/inputs'
 reassemble(input_root,archives,manifest_sha256=PIN)
 runtime=work/'runtime-source'; skill=work/'skill-source'
 extract(archives/'runtime.zip',runtime)
 extract(archives/'locomotion-input.zip',skill)
 check_rows(runtime,plan['runtime_files']);check_rows(skill,plan['skill_files'])
 # Keep each archive's manifest separate. Merge only the bundle's disjoint data.
 import shutil
 payload=work/'payload';payload.mkdir()
 for name in ['bundle','native-source','deps']:shutil.copytree(runtime/name,payload/name)
 for source in (skill/'bundle').rglob('*'):
  if source.is_file():
   dest=payload/'bundle'/source.relative_to(skill/'bundle')
   if dest.exists():raise ValueError('bundle_member_collision')
   dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 shutil.copytree(skill/'teacher-code',payload/'teacher-code')
 for name in ['config.json','teacher_child.py','teacher_algo.py','speed10_audit.py','resume_audit.py']:
  (payload/'teacher-code'/name).write_bytes((ROOT/'overrides'/name).read_bytes())
 (payload/'bundle/overlay/wbt_training/config_values/hud04_robot.py').write_bytes((ROOT/'overrides/hud04_robot.py').read_bytes())
 (payload/'bundle/asset/urdf/HU_D04_01.urdf').write_bytes((ROOT/'overrides/HU_D04_01.urdf').read_bytes())
 check_rows(payload,plan['effective_overrides'])
 validate_config(json.loads((payload/'teacher-code/config.json').read_text()))
 reassemble(ROOT/'checkpoint-shards',work/'checkpoint-input',manifest_sha256=CHECKPOINT_PIN)
 cp=work/'checkpoint-input/model_00500.pt'
 assert cp.is_file() and not cp.is_symlink() and cp.stat().st_size==10344063 and sha(cp)==SOURCE_SHA
 emit('git_checkpoint_reassembled',bytes=cp.stat().st_size,sha256=SOURCE_SHA,source_completed_updates=501)
 return payload

def learning_report(out,code):
 r=dict(native_returncode=code,completed_updates=0,learning_evidence_passed=False,
        native_lifecycle_passed=False,worker_passed=False,teacher_converged=False)
 try:
  d=json.loads((out/'learning_state.json').read_text());r['completed_updates']=d['completed_updates']
  cp=out/d['checkpoint_relative_path']
  flags=['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite']
  r['learning_evidence_passed']=(1<=d['completed_updates']<=UPDATES and d.get('new_process_resume_verified') is True and all(d.get(k) is True for k in flags)
    and cp.is_file() and not cp.is_symlink() and cp.resolve().is_relative_to(out.resolve()) and sha(cp)==d['checkpoint_sha256'])
  life=json.loads((out/'lifecycle.json').read_text())
  r['native_lifecycle_passed']=code==0 and life.get('close_returned') is True and life.get('child_returned') is True
  r['worker_passed']=r['learning_evidence_passed'] and r['native_lifecycle_passed']
 except (OSError,ValueError,KeyError,TypeError):pass
 return r

def seal(out,work,report):
 selected=[]
 for p in sorted(out.rglob('*')):
  if p.is_symlink():continue
  if p.is_file() and (p.suffix in {'.json','.jsonl','.yaml','.yml'} or p.name in {'model_smoke_final.pt','model_at_672.pt'}):
   selected.append(p)
 # Preserve latest newly trained checkpoint even if native teardown failed.
 if not any(p.name=='model_smoke_final.pt' for p in selected):
  matches=[p for p in out.rglob('model_latest.pt') if p.is_file() and not p.is_symlink()]
  if len(matches)==1:selected.append(matches[0])
 if sum(p.stat().st_size for p in selected)>64*1024**2:raise ValueError('result_size_cap')
 manifest=dict(report=report,files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in selected],
               private_task_log_mirror=True,remote_result_retrieval_verified=False)
 (work/'result-manifest.json').write_text(json.dumps(manifest,indent=2))
 archive=work/'locomotion-result.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  z.write(work/'result-manifest.json','result-manifest.json')
  for p in selected:z.write(p,str(p.relative_to(out)))
 # A task-private, credential-free fallback, not a public Git upload or a claim
 # that Flux's small preview retains the full archive. Full-log readback needed.
 emit('result_archive_begin',bytes=archive.stat().st_size,sha256=sha(archive),chunk_bytes=12288)
 with archive.open('rb') as f:
  i=0
  for b in iter(lambda:f.read(12288),b''):
   emit('result_archive_chunk',index=i,data=base64.b64encode(b).decode('ascii'));i+=1
 emit('result_archive_end',chunks=i,bytes=archive.stat().st_size,sha256=sha(archive),remote_retrieval_verified=False)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
 p.add_argument('--task-id');p.add_argument('--work',type=Path);p.add_argument('--commit',required=True)
 a=p.parse_args();plan=json.loads((ROOT/'PLAN.json').read_text());check_rows(ROOT,plan['package_files'])
 if not re.fullmatch('[0-9a-f]{40}',a.commit):raise ValueError('commit_pin_required')
 if a.execute:
  actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
  if actual!=a.commit:raise ValueError('remote_commit_mismatch')
 if a.execute and (not a.task_id or not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id)):raise ValueError('task_id_required')
 work=a.work or ROOT.parent/('fresh-locomotion-'+(a.task_id or 'cpu-preflight'))
 work.mkdir(parents=True,exist_ok=False);started=time.monotonic();child=None
 def deadline(signum,frame):raise TimeoutError('worker_budget_reached')
 if a.execute:
  signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.setitimer(signal.ITIMER_REAL,LIMIT)
 out=work/'output';out.mkdir()
 report=dict(task_id=a.task_id,git_commit=a.commit,fresh_training=False,skill='locomotion',worker_seconds_max=LIMIT,
             update_cap=UPDATES,num_envs=4096,clips=100,frames=71042,automatic_retries=0,worker_passed=False,teacher_converged=False)
 try:
  payload=prepare(work,plan)
  emit('inputs_verified',fresh_training=False,num_envs=4096,clips=100,frames=71042,source_checkpoint_sha256=SOURCE_SHA)
  if not a.execute:
   emit('cpu_preflight_passed',gpu_started=False,**report);return 0
  cp=work/'checkpoint-input/model_00500.pt'
  assert cp.is_file() and not cp.is_symlink() and cp.stat().st_size==10344063 and sha(cp)==SOURCE_SHA
  emit('git_resume_checkpoint_verified',bytes=cp.stat().st_size,sha256=SOURCE_SHA,source_completed_updates=501)
  vendor=next((Path(x) for x in ['/isaac-sim/python.sh','/workspace/isaaclab/_isaac_sim/python.sh'] if Path(x).is_file()),None)
  if vendor is None:raise ValueError('vendor_python_missing')
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'])}
  for k in ['flux_key1103','php1021','ins1116','limxanzhuangbao','tian1119tian','sxsc_tushare_token','jhiu54']:env.pop(k,None)
  env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',PHP_FLUX_TASK_ID=a.task_id,PHP_RESUME_CHECKPOINT=str(cp))
  site=work/'dependency-site'
  command=[str(vendor),'-m','pip','install','--no-index','--no-deps','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')]
  with (out/'install.log').open('xb') as log:
   code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(240,max(1,started+LIMIT-300-time.monotonic()))).returncode
  if code:raise RuntimeError('offline_dependency_install_failed')
  command=[str(vendor),str(payload/'teacher-code/teacher_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),
    '--thread-id',plan['provenance']['thread_id'],'--turn-id',plan['provenance']['turn_id']]
  emit('simulator_start_requested',checkpoint_load_requested=True,update_cap=UPDATES,source_completed_updates=501)
  with (out/'native.log').open('xb') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   sent=0;evidence_sent=set()
   while child.poll() is None:
    elapsed=time.monotonic()-started
    if elapsed>=1500:(out/'STOP_AFTER_UPDATE').touch(exist_ok=True)
    if elapsed>=1670:raise TimeoutError('native_child_deadline')
    for evidence_name in ['resume-state.json','native-velocity-limits.json','capacity.json']:
     evidence_path=out/evidence_name
     if evidence_name not in evidence_sent and evidence_path.is_file():
      try:proof=json.loads(evidence_path.read_text())
      except json.JSONDecodeError:continue
      event=proof.pop('event');assert event in {'resume_state_verified','native_velocity_limits_verified','capacity_initialized'}
      emit(event,**proof);evidence_sent.add(evidence_name)
    progress=out/'updates.jsonl'
    if progress.is_file():
     with progress.open() as f:
      f.seek(sent);lines=f.readlines()
      for line in lines:
       if line.endswith('\n'):
        emit('training_progress',**json.loads(line));sent+=len(line.encode())
    time.sleep(2)
  report.update(learning_report(out,child.returncode))
 except BaseException as e:
  report['failure_type']=type(e).__name__
  # No arbitrary exception string or potentially secret-bearing raw log.
  import traceback
  report['failure_frames']=[dict(file=Path(x.filename).name,function=x.name,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)][-8:]
 finally:
  if child is not None and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=5)
  if a.execute:
   report['elapsed_seconds']=round(time.monotonic()-started,2)
   (out/'report.json').write_text(json.dumps(report,indent=2))
   # Archive includes only selected local evidence and new checkpoint, never
   # raw install/native logs, environment values, or original asset bundles.
   seal(out,work,report);emit('worker_finished',**report,remote_result_retrieval_verified=False)
   signal.setitimer(signal.ITIMER_REAL,0)
 return 0 if report['worker_passed'] else 1

if __name__=='__main__':sys.exit(main())