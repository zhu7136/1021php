"""One authorized G1 low-climb continuation, 1800s worker, zero retries.

CPU checks never start the simulator. --execute requires exact Git commit.
Reuses hash-verified original full training inputs; loads only teacher086.pt.
"""
import argparse,hashlib,importlib.util,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'low-climb-g1-fresh-20261122'
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('original_fresh_worker',SOURCE/'train_fresh.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
LIMIT=1800
UPDATES=2000
SOURCE_UPDATES=650
SOURCE_SHA='ec7d0fb9e7e10e8758a0d8936680a28508e9cda778157904b8f56b34c6f4e18e'
sha=base.sha
emit=base.emit

def validate_config(c):
 original=json.loads((SOURCE/'overrides/config.json').read_text())
 effective=json.loads(json.dumps(c))
 assert c['training']['checkpoint']=='/workspace/resume-input/model_smoke_final.pt'
 assert c['training']['name']=='hud04-low-climb-G1-RESUME650-30min-20261123'
 assert c['algo']['config']['load_optimizer'] is True
 assert c['algo']['config']['num_learning_iterations']==UPDATES
 for k in ['name','checkpoint']:effective['training'][k]=original['training'][k]
 assert effective==original,'unexpected_training_contract_change'
 return True

def validate_package():
 plan=json.loads((ROOT/'PLAN.json').read_text())
 base.check_rows(ROOT,plan['package_files'])
 old=json.loads((SOURCE/'PLAN.json').read_text())
 base.check_rows(SOURCE,old['package_files'])
 validate_config(json.loads((ROOT/'overrides/config.json').read_text()))
 return plan,old

def prepare(work,plan,old):
 payload=base.prepare(work,old)
 checkpoint_dir=work/'checkpoint-input'
 manifest=ROOT.parent/'low-climb086-g1-native-20261122/inputs/shard-manifest.json'
 assert sha(manifest)==plan['checkpoint_manifest_sha256']
 rows=json.loads(manifest.read_text())['files']
 row=next(x for x in rows if x['name']=='teacher086.pt')
 assert row['sha256']==SOURCE_SHA and row['bytes']==10344251
 checkpoint_dir.mkdir()
 cp=checkpoint_dir/'teacher086.pt'
 with cp.open('xb') as out:
  for part in row['shards']:
   assert re.fullmatch('f00-p[0-9]{5}.bin',part['path'])
   p=manifest.parent/part['path']
   assert p.is_file() and not p.is_symlink()
   assert p.stat().st_size==part['bytes'] and sha(p)==part['sha256']
   with p.open('rb') as f:shutil.copyfileobj(f,out)
 assert cp.stat().st_size==row['bytes'] and sha(cp)==SOURCE_SHA
 for name in ['config.json','teacher_child.py','teacher_algo.py','resume_audit.py']:
  (payload/'teacher-code'/name).write_bytes((ROOT/'overrides'/name).read_bytes())
 validate_config(json.loads((payload/'teacher-code/config.json').read_text()))
 return payload,cp

def learning_report(out,code):
 r=base.learning_report(out,code)
 try:
  loaded=json.loads((out/'resume-state.json').read_text())
  trained=json.loads((out/'learning_state.json').read_text())
  r['native_resume_verified']=(loaded['exact_training_state_loaded'] is True and loaded['source_checkpoint_sha256']==SOURCE_SHA and loaded['source_completed_updates']==SOURCE_UPDATES)
  r['learning_evidence_passed']=r['learning_evidence_passed'] and r['native_resume_verified'] and trained['new_process_resume_verified'] is True
  r['cumulative_completed_updates']=SOURCE_UPDATES+trained['completed_updates']
 except (OSError,ValueError,KeyError,TypeError):
  r['native_resume_verified']=False;r['learning_evidence_passed']=False
 r['worker_passed']=r['learning_evidence_passed'] and r['native_lifecycle_passed']
 return r

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--execute',action='store_true');parser.add_argument('--commit',required=True)
 parser.add_argument('--task-id');parser.add_argument('--work',type=Path)
 a=parser.parse_args();plan,old=validate_package()
 assert re.fullmatch('[0-9a-f]{40}',a.commit)
 if not a.execute:
  emit('cpu_package_checks_passed',fresh_training=False,checkpoint_sha256=SOURCE_SHA,worker_seconds_max=LIMIT,gpu_started=False)
  return 0
 assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==a.commit
 assert a.task_id and re.fullmatch(r'TASK_\d{8}_\d+',a.task_id)
 work=a.work or ROOT.parent/('resume650-'+a.task_id)
 work.mkdir(parents=True,exist_ok=False);out=work/'output';out.mkdir()
 started=time.monotonic();child=None
 def deadline(signum,frame):raise TimeoutError('worker_budget_reached')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline)
 signal.setitimer(signal.ITIMER_REAL,LIMIT)
 report=dict(task_id=a.task_id,git_commit=a.commit,fresh_training=False,skill='low-climb',source_task='TASK_20261122_086',source_completed_updates=650,source_checkpoint_sha256=SOURCE_SHA,worker_seconds_max=LIMIT,update_cap=UPDATES,num_envs=4096,clips=40,frames=32814,automatic_retries=0,worker_passed=False,teacher_converged=False)
 try:
  payload,cp=prepare(work,plan,old)
  emit('inputs_verified',fresh_training=False,num_envs=4096,clips=40,frames=32814,checkpoint_sha256=SOURCE_SHA)
  vendor=next((Path(x) for x in ['/isaac-sim/python.sh','/workspace/isaaclab/_isaac_sim/python.sh'] if Path(x).is_file()),None)
  if vendor is None:raise ValueError('vendor_python_missing')
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'])}
  env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',PHP_FLUX_TASK_ID=a.task_id,PHP_RESUME_CHECKPOINT=str(cp))
  site=work/'dependency-site'
  command=[str(vendor),'-m','pip','install','--no-index','--no-deps','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')]
  with (out/'install.log').open('xb') as log:
   code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(240,max(1,started+LIMIT-300-time.monotonic()))).returncode
  if code:raise RuntimeError('offline_dependency_install_failed')
  command=[str(vendor),str(payload/'teacher-code/teacher_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--thread-id',plan['provenance']['thread_id'],'--turn-id',plan['provenance']['turn_id']]
  emit('simulator_start_requested',checkpoint_load_requested=True,source_completed_updates=650,update_cap=UPDATES)
  with (out/'native.log').open('xb') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   sent=0;sent_evidence=set()
   while child.poll() is None:
    elapsed=time.monotonic()-started
    if elapsed>=1500:(out/'STOP_AFTER_UPDATE').touch(exist_ok=True)
    if elapsed>=1670:raise TimeoutError('native_child_deadline')
    for evidence in ['native-velocity-limits.json','resume-state.json','capacity.json']:
     ep=out/evidence
     if evidence not in sent_evidence and ep.is_file():
      try:
       ev=json.loads(ep.read_text());event=ev.pop('event');emit(event,**ev);sent_evidence.add(evidence)
      except ValueError:pass
    progress=out/'updates.jsonl'
    if progress.is_file():
     with progress.open() as f:
      f.seek(sent)
      for line in f:
       if line.endswith('\n'):emit('training_progress',**json.loads(line));sent+=len(line.encode())
    time.sleep(2)
  report.update(learning_report(out,child.returncode))
 except BaseException as e:
  import traceback
  report['failure_type']=type(e).__name__
  report['failure_frames']=[dict(file=Path(x.filename).name,function=x.name,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)][-8:]
 finally:
  if child is not None and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=5)
  report['elapsed_seconds']=round(time.monotonic()-started,2)
  (out/'report.json').write_text(json.dumps(report,indent=2))
  base.seal(out,work,report);emit('worker_finished',**report,remote_result_retrieval_verified=False)
  signal.setitimer(signal.ITIMER_REAL,0)
 return 0 if report['worker_passed'] else 1

if __name__=='__main__':sys.exit(main())