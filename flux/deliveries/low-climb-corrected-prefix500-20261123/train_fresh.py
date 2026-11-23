"""One fresh low-climb teacher, offline inputs, bounded worker, zero retries.

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
UPDATES=200

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
 assert not any(t.get(k) for k in ['checkpoint','teacher_checkpoint','registry_name','wandb_path'])
 assert a['num_learning_iterations']==UPDATES and a['num_steps_per_env']==24
 assert len(c['robot']['dof_names'])==31 and c['robot']['actions_dim']==29
 assert 'low-climb' in t['name']
 original=json.loads((ROOT/'original-config.json').read_text())
 comparison=json.loads(json.dumps(c))
 for k in ['name','checkpoint','teacher_checkpoint','registry_name','wandb_path']:comparison['training'][k]=original['training'].get(k)
 comparison['algo']['config']['num_learning_iterations']=original['algo']['config']['num_learning_iterations']
 comparison['robot']['dof_vel_limit_list']=original['robot']['dof_vel_limit_list']
 assert comparison==original, 'unapproved_training_contract_change'
 from overrides.speed10_audit import G1_LIMITS
 assert c['robot']['dof_vel_limit_list']==[G1_LIMITS.get(n,v) for n,v in zip(original['robot']['dof_names'],original['robot']['dof_vel_limit_list'])], 'g1_limit_mapping_mismatch'
 assert c['randomization']['setup_terms']['setup_dof_pos_bias']['params']['enabled'] is False

def prepare(work,plan):
 import shutil
 own=work/'training-archives'
 source=ROOT.parent/plan['source_training_delivery']
 reassemble(source/'inputs',own,manifest_sha256=plan['source_training_manifest_sha256'])
 assert {p.name for p in own.iterdir()}=={'full-training-input.zip','reassembly-receipt.json'}, 'unexpected_training_archive'
 shared=ROOT.parent/plan['shared_runtime_inputs']
 manifest=shared/'shard-manifest.json';assert sha(manifest)==plan['shared_runtime_manifest_sha256']
 row=plan['shared_runtime_file'];assert json.loads(manifest.read_text())['files'][0]==row
 runtime=work/'runtime.zip'
 with runtime.open('xb') as out:
  for chunk in row['shards']:
   part=shared/chunk['path'];assert re.fullmatch('f00-p[0-9]{5}.bin',chunk['path']) and not part.is_symlink()
   assert part.stat().st_size==chunk['bytes'] and sha(part)==chunk['sha256']
   with part.open('rb') as f:shutil.copyfileobj(f,out)
 assert runtime.stat().st_size==row['bytes'] and sha(runtime)==row['sha256']
 payload=work/'payload';extract(runtime,payload);check_rows(payload,plan['runtime_files'])
 extract(own/'full-training-input.zip',payload);check_rows(payload,plan['original_input_files'])
 for name in ['config.json','teacher_child.py','teacher_algo.py','speed10_audit.py']:
  (payload/'teacher-code'/name).write_bytes((ROOT/'overrides'/name).read_bytes())
 (payload/'bundle/overlay/wbt_training/config_values/hud04_robot.py').write_bytes((ROOT/'overrides/hud04_robot.py').read_bytes())
 (payload/'bundle/asset/urdf/HU_D04_01.urdf').write_bytes((ROOT/'overrides/HU_D04_01.urdf').read_bytes())
 check_rows(payload,plan['effective_overrides'])
 shutil.copyfile(ROOT/'corrected-reference.npz',payload/'bundle/reference/full-reference.npz')
 manifest=json.loads((ROOT/'training-reference-manifest.json').read_text())
 assert manifest['ready_for_bounded_native_gpu_probe'] and manifest['prefix_only'] and manifest['frames']==500
 assert sha(payload/'bundle/reference/full-reference.npz')==manifest['training_reference_sha256']
 shutil.copyfile(ROOT/'overrides/probe_audit.py',payload/'teacher-code/probe_audit.py')
 shutil.copyfile(ROOT/'training-reference-manifest.json',payload/'teacher-code/training-reference-manifest.json')
 import numpy as np
 with np.load(payload/'bundle/reference/full-reference.npz') as ref:
  assert ref['joint_pos'].shape==(500,38) and int(ref['motion_ends'].sum())==1 and ref['motion_ends'][-1]
  assert int(ref['fps'][0])==50 and int(ref['motion_idxs'].max())==0
  for name in ref.files:
   if ref[name].dtype.kind in 'fc':assert np.isfinite(ref[name]).all()
 validate_config(json.loads((payload/'teacher-code/config.json').read_text()))
 assert not any(p.suffix.lower() in {'.pt','.pth','.ckpt','.onnx','.safetensors'} for p in payload.rglob('*')), 'historical_weights_present'
 return payload

def learning_report(out,code):
 r=dict(native_returncode=code,completed_updates=0,learning_evidence_passed=False,
        native_lifecycle_passed=False,worker_passed=False,teacher_converged=False)
 try:
  d=json.loads((out/'learning_state.json').read_text());r['completed_updates']=d['completed_updates']
  cp=out/d['checkpoint_relative_path']
  flags=['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite']
  r['learning_evidence_passed']=(1<=d['completed_updates']<=UPDATES and all(d.get(k) is True for k in flags)
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
 archive=work/'low-climb-result.zip'
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
 work=a.work or ROOT.parent/('fresh-low-climb-'+(a.task_id or 'cpu-preflight'))
 work.mkdir(parents=True,exist_ok=False);started=time.monotonic();child=None
 def deadline(signum,frame):raise TimeoutError('worker_budget_reached')
 if a.execute:
  signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.setitimer(signal.ITIMER_REAL,LIMIT)
 out=work/'output';out.mkdir()
 report=dict(task_id=a.task_id,git_commit=a.commit,fresh_training=True,skill='low-climb',worker_seconds_max=LIMIT,
             update_cap=UPDATES,num_envs=4096,clips=1,frames=500,automatic_retries=0,worker_passed=False,teacher_converged=False)
 try:
  payload=prepare(work,plan)
  emit('inputs_verified',fresh_training=True,num_envs=4096,clips=1,frames=500,historical_checkpoint_loaded=False)
  if not a.execute:
   emit('cpu_preflight_passed',gpu_started=False,**report);return 0
  vendor=next((Path(x) for x in ['/isaac-sim/python.sh','/workspace/isaaclab/_isaac_sim/python.sh'] if Path(x).is_file()),None)
  if vendor is None:raise ValueError('vendor_python_missing')
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'])}
  for k in ['jhiu54','flux_key1103','php1021','ins1116','limxanzhuangbao','tian1119tian','sxsc_tushare_token']:env.pop(k,None)
  env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',PHP_FLUX_TASK_ID=a.task_id)
  site=work/'dependency-site'
  command=[str(vendor),'-m','pip','install','--no-index','--no-deps','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')]
  with (out/'install.log').open('xb') as log:
   code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(240,max(1,started+LIMIT-300-time.monotonic()))).returncode
  if code:raise RuntimeError('offline_dependency_install_failed')
  command=[str(vendor),str(payload/'teacher-code/teacher_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),
    '--thread-id',plan['provenance']['thread_id'],'--turn-id',plan['provenance']['turn_id']]
  emit('simulator_start_requested',checkpoint_load_requested=False,update_cap=UPDATES)
  with (out/'native.log').open('xb') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   sent=0;sent_evidence=set()
   while child.poll() is None:
    elapsed=time.monotonic()-started
    if elapsed>=1500:(out/'STOP_AFTER_UPDATE').touch(exist_ok=True)
    if elapsed>=1670:raise TimeoutError('native_child_deadline')
    for evidence in ['native-velocity-limits.json','fresh-initialization.json','capacity.json','native-corrected-reference.json']:
     ep=out/evidence
     if evidence not in sent_evidence and ep.is_file():
      try:
       ev=json.loads(ep.read_text());event=ev.pop('event');emit(event,**ev);sent_evidence.add(evidence)
      except ValueError:pass
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