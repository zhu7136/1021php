"""Git-only, offline, one-GPU frozen native comparison. Default CPU preflight."""
import argparse,base64,hashlib,json,os,re,shutil,signal,subprocess,sys,time,traceback,zipfile
from pathlib import Path,PurePosixPath
from local_shards import reassemble
ROOT=Path(__file__).resolve().parent
LIMIT=900
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def emit(event,**data):print(json.dumps(dict(event=event,**data),allow_nan=False),flush=True)
def extract(path,target):
 with zipfile.ZipFile(path) as z:
  seen=set()
  for i in z.infolist():
   p=PurePosixPath(i.filename)
   if p.is_absolute() or '..' in p.parts or '\\' in i.filename or i.filename in seen or i.file_size>512*1024**2:raise ValueError('unsafe_zip')
   if ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('symlink_in_zip')
   if p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.onnx'}:raise ValueError('unexpected_weight_in_runtime_archive')
   seen.add(i.filename)
  z.extractall(target)
def prepare(work,plan,shared_override=None):
 own=work/'own-archives';reassemble(ROOT/'inputs',own,manifest_sha256=plan['inputs_manifest_sha256'])
 cp=own/'teacher086.pt';assert cp.stat().st_size==plan['checkpoint']['bytes'] and sha(cp)==plan['checkpoint']['sha256']
 shared=shared_override or ROOT.parent/plan['shared_runtime_inputs'];manifest=shared/'shard-manifest.json'
 assert sha(manifest)==plan['shared_runtime_manifest_sha256']
 row=plan['shared_runtime_file'];assert json.loads(manifest.read_text())['files'][0]==row
 runtime=work/'runtime.zip'
 with runtime.open('xb') as out:
  for chunk in row['shards']:
   part=shared/chunk['path'];assert re.fullmatch('f00-p[0-9]{5}.bin',chunk['path']) and not part.is_symlink()
   assert part.stat().st_size==chunk['bytes'] and sha(part)==chunk['sha256']
   with part.open('rb') as f:shutil.copyfileobj(f,out)
 assert runtime.stat().st_size==row['bytes'] and sha(runtime)==row['sha256']
 payload=work/'payload';extract(runtime,payload)
 for item in plan['runtime_files']:
  q=payload/item['path'];assert not q.is_symlink() and q.stat().st_size==item['bytes'] and sha(q)==item['sha256']
 extract(own/'single-clip-input.zip',payload)
 for name,dest in [('hud04_robot.py','bundle/overlay/wbt_training/config_values/hud04_robot.py'),('HU_D04_01.urdf','bundle/asset/urdf/HU_D04_01.urdf')]:
  shutil.copy2(ROOT/'overrides'/name,payload/dest)
  assert sha(payload/dest)==sha(ROOT/'overrides'/name)
 config=json.loads((payload/'teacher-code/config.json').read_text())
 assert config['training']['num_envs']==1 and config['training']['checkpoint'] is None
 assert not config['randomization']['setup_terms'] and not config['randomization']['step_terms'] and not config['randomization']['reset_terms']
 assert not any(x['enable_noise'] for x in config['observation']['groups'].values())
 assert config['command']['setup_terms']['motion_command']['params']['motion_config']['resample_on_motion_end'] is False
 return payload,cp
def seal(out,work,report):
 selected=[p for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.jsonl','.log'}]
 assert sum(x.stat().st_size for x in selected)<64*1024**2
 manifest=dict(report=report,files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in selected],remote_result_retrieval_verified=False)
 archive=work/'frozen-native-result.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  z.writestr('result-manifest.json',json.dumps(manifest,indent=2))
  for p in selected:z.write(p,str(p.relative_to(out)))
 emit('result_archive_begin',bytes=archive.stat().st_size,sha256=sha(archive),chunk_bytes=12288)
 with archive.open('rb') as f:
  i=0
  for data in iter(lambda:f.read(12288),b''):
   emit('result_archive_chunk',index=i,data=base64.b64encode(data).decode('ascii'));i+=1
 emit('result_archive_end',chunks=i,bytes=archive.stat().st_size,sha256=sha(archive),remote_retrieval_verified=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--commit',required=True);p.add_argument('--task-id');p.add_argument('--work',type=Path,required=True);p.add_argument('--shared-input-root',type=Path);a=p.parse_args()
 plan=json.loads((ROOT/'PLAN.json').read_text());assert plan['scope']['worker_seconds_max']==900 and plan['scope']['training_updates']==0
 for row in plan['package_files']:
  q=ROOT/row['path'];assert not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
 if not re.fullmatch('[0-9a-f]{40}',a.commit):raise ValueError('commit_required')
 if a.execute:
  assert a.shared_input_root is None,'runtime overrides are CPU-preflight-only'
  assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==a.commit
  assert re.fullmatch(r'TASK_\d{8}_\d+',a.task_id or '')
 work=a.work;work.mkdir(parents=True,exist_ok=False);out=work/'output';out.mkdir();start=time.monotonic();child=None
 def deadline(signum,frame):raise TimeoutError('worker_900s_deadline')
 if a.execute:
  signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.setitimer(signal.ITIMER_REAL,LIMIT)
 report=dict(task_id=a.task_id,git_commit=a.commit,worker_seconds_max=900,training_updates=0,automatic_retries=0,worker_passed=False,native_execution_started=False,checkpoint_sha256=plan['checkpoint']['sha256'])
 try:
  payload,cp=prepare(work,plan,a.shared_input_root);emit('git_shards_reassembled_and_verified',checkpoint_sha256=sha(cp),checkpoint_iteration=649,reference_frames=800)
  if not a.execute:emit('cpu_package_preflight_passed',gpu_tasks_started=0);return 0
  vendor=next((Path(x) for x in ['/isaac-sim/python.sh','/workspace/isaaclab/_isaac_sim/python.sh'] if Path(x).is_file()),None)
  if vendor is None:raise RuntimeError('vendor_python_missing')
  env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'])}
  for k in ['jhiu54','php1021','flux_key1103','ins1116','limxanzhuangbao','tian1119tian','sxsc_tushare_token']:env.pop(k,None)
  env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0')
  site=work/'dependency-site';cmd=[str(vendor),'-m','pip','install','--no-index','--no-deps','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')]
  with (out/'install.log').open('xb') as log:code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(200,max(1,760-(time.monotonic()-start)))).returncode
  if code:raise RuntimeError('offline_install_failed')
  cmd=[str(vendor),str(ROOT/'native_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--checkpoint',str(cp)]
  with (out/'native.log').open('xb') as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);report['native_execution_started']=True;emit('frozen_native_child_started',training_updates=0,max_policy_steps=799)
   while child.poll() is None:
    if time.monotonic()-start>=800:raise TimeoutError('child_deadline_reserving_result_seal_time')
    time.sleep(2)
  report['native_returncode']=child.returncode
  if (out/'native-result.json').is_file():
   result=json.loads((out/'native-result.json').read_text());report.update({k:result[k] for k in ['outcome','policy_steps','simulated_seconds','weights_unchanged','full_reference_clip_reached']})
   report['native_evidence_written']=True
  report['worker_passed']=child.returncode==0 and report.get('native_evidence_written',False)
 except BaseException as e:
  report.update(failure_type=type(e).__name__,failure_frames=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)][-8:])
  emit('worker_preflight_or_execution_failed',**report)
 finally:
  if child is not None and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=5)
  if a.execute:
   report['elapsed_seconds']=round(time.monotonic()-start,2);(out/'report.json').write_text(json.dumps(report,indent=2));seal(out,work,report)
   emit('worker_finished',**report,remote_result_retrieval_verified=False);signal.setitimer(signal.ITIMER_REAL,0)
 return 0 if report['worker_passed'] else 1
if __name__=='__main__':sys.exit(main())