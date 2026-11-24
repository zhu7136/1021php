"""One GPU continuation: real learning3600s, worker4200s, zero automatic retries."""
import argparse,importlib.util,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'low-climb-corrected-prefix500-20261123'
sys.path.insert(0,str(SOURCE));spec=importlib.util.spec_from_file_location('previous_worker',SOURCE/'train_fresh.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
LIMIT=4200;TRAIN_SECONDS=3600;UPDATES=100000;SOURCE_UPDATES=200;SOURCE_SHA='370df740c687140ec8b87de7ae70bb1f9d3f2b88ed8983a0ab9f38e8fb0bc09b'
sha=base.sha;emit=base.emit
def validate_config(c):
 original=json.loads((SOURCE/'overrides/config.json').read_text());effective=json.loads(json.dumps(c))
 assert c['training']['checkpoint']=='/workspace/resume-input/model_source.pt' and c['algo']['config']['load_optimizer']
 assert c['algo']['config']['num_learning_iterations']==UPDATES and c['training']['name']=='hud04-low-climb-corrected-prefix500-RESUME200-train60min-20261124'
 for k in ['name','checkpoint']:effective['training'][k]=original['training'][k]
 effective['algo']['config']['num_learning_iterations']=original['algo']['config']['num_learning_iterations'];assert effective==original,'unexpected_training_contract_change'
def validate_package():
 plan=json.loads((ROOT/'PLAN.json').read_text());base.check_rows(ROOT,plan['package_files']);old=json.loads((SOURCE/'PLAN.json').read_text());base.check_rows(SOURCE,old['package_files']);validate_config(json.loads((ROOT/'config.json').read_text()))
 assert sha(ROOT/'model_source.pt')==SOURCE_SHA;assert plan['training_target_seconds']==TRAIN_SECONDS and plan['worker_seconds_max']==LIMIT
 return plan,old
def prepare(work,old):
 payload=base.prepare(work,old)
 for name in ['config.json','teacher_child.py','teacher_algo.py','resume_audit.py','probe_audit.py']:shutil.copyfile(ROOT/name,payload/'teacher-code'/name)
 cp=work/'checkpoint-input/model_source.pt';cp.parent.mkdir();shutil.copyfile(ROOT/'model_source.pt',cp);assert sha(cp)==SOURCE_SHA
 validate_config(json.loads((payload/'teacher-code/config.json').read_text()));return payload,cp
def learning_report(out,code):
 r=dict(native_returncode=code,completed_updates=0,learning_evidence_passed=False,native_resume_verified=False,native_lifecycle_passed=False,worker_passed=False,teacher_converged=False,training_target_met=False)
 try:
  d=json.loads((out/'learning_state.json').read_text());loaded=json.loads((out/'resume-state.json').read_text());life=json.loads((out/'lifecycle.json').read_text());cp=out/d['checkpoint_relative_path']
  r.update(completed_updates=d['completed_updates'],cumulative_completed_updates=SOURCE_UPDATES+d['completed_updates'],training_elapsed_seconds=d['training_elapsed_seconds'],stop_reason=d['stop_reason'])
  r['native_resume_verified']=loaded['exact_training_state_loaded'] and loaded['source_checkpoint_sha256']==SOURCE_SHA and loaded['source_completed_updates']==SOURCE_UPDATES
  r['learning_evidence_passed']=r['native_resume_verified'] and 1<=d['completed_updates']<=UPDATES and all(d.get(k) is True for k in ['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite','new_process_resume_verified']) and cp.is_file() and not cp.is_symlink() and cp.resolve().is_relative_to(out.resolve()) and sha(cp)==d['checkpoint_sha256']
  r['training_target_met']=d['stop_reason']=='training_duration_reached' and d['training_elapsed_seconds']>=TRAIN_SECONDS
  r['native_lifecycle_passed']=code==0 and life.get('close_returned') is True and life.get('child_returned') is True
  r['worker_passed']=r['learning_evidence_passed'] and r['training_target_met'] and r['native_lifecycle_passed']
 except (OSError,ValueError,KeyError,TypeError):pass
 return r
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--commit',required=True);p.add_argument('--task-id');p.add_argument('--work',type=Path);a=p.parse_args();plan,old=validate_package();assert re.fullmatch('[0-9a-f]{40}',a.commit)
 if not a.execute:
  emit('cpu_package_checks_passed',gpu_started=False,fresh_training=False,source_checkpoint_sha256=SOURCE_SHA,training_target_seconds=TRAIN_SECONDS,worker_seconds_max=LIMIT,maximum_additional_updates=UPDATES);return 0
 assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==a.commit
 assert a.task_id and re.fullmatch(r'TASK_\d{8}_\d+',a.task_id)
 work=a.work or ROOT.parent/('corrected-resume200-hour-'+a.task_id);work.mkdir(parents=True,exist_ok=False);out=work/'output';out.mkdir();started=time.monotonic();child=None
 def deadline(signum,frame):raise TimeoutError('worker_budget_reached')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.setitimer(signal.ITIMER_REAL,LIMIT)
 report=dict(task_id=a.task_id,git_commit=a.commit,fresh_training=False,skill='low-climb',source_task='TASK_20261123_410',source_checkpoint_sha256=SOURCE_SHA,source_completed_updates=SOURCE_UPDATES,training_target_seconds=TRAIN_SECONDS,worker_seconds_max=LIMIT,maximum_additional_updates=UPDATES,num_envs=4096,clips=1,frames=500,automatic_retries=0,worker_passed=False,teacher_converged=False)
 try:
  payload,cp=prepare(work,old);emit('inputs_verified',checkpoint_sha256=sha(cp),fresh_training=False,clips=1,frames=500)
  vendor=next((Path(x) for x in ['/isaac-sim/python.sh','/workspace/isaaclab/_isaac_sim/python.sh'] if Path(x).is_file()),None);assert vendor is not None,'vendor_python_missing'
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'])}
  for k in ['jhiu54','flux_key1103','php1021','ins1116','limxanzhuangbao','tian1119tian','sxsc_tushare_token']:env.pop(k,None)
  env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',PHP_FLUX_TASK_ID=a.task_id,PHP_RESUME_CHECKPOINT=str(cp))
  site=work/'dependency-site';command=[str(vendor),'-m','pip','install','--no-index','--no-deps','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')]
  with (out/'install.log').open('xb') as log:code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
  if code:raise RuntimeError('offline_dependency_install_failed')
  command=[str(vendor),str(payload/'teacher-code/teacher_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--thread-id',plan['provenance']['thread_id'],'--turn-id',plan['provenance']['turn_id']]
  emit('simulator_start_requested',checkpoint_load_requested=True,training_target_seconds=TRAIN_SECONDS,maximum_additional_updates=UPDATES)
  with (out/'native.log').open('xb') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);sent=0;seen=set()
   while child.poll() is None:
    elapsed=time.monotonic()-started
    if elapsed>=3960:(out/'STOP_AFTER_UPDATE').touch(exist_ok=True)
    if elapsed>=4080:raise TimeoutError('native_child_deadline')
    for name in ['resume-state.json','native-corrected-reference.json','native-velocity-limits.json','training-timer.json','capacity.json']:
     path=out/name
     if name not in seen and path.is_file():
      try:e=json.loads(path.read_text());event=e.pop('event');emit(event,**e);seen.add(name)
      except ValueError:pass
    progress=out/'updates.jsonl'
    if progress.is_file():
     with progress.open() as f:
      f.seek(sent)
      for line in f.readlines():
       if line.endswith('\n'):emit('training_progress',**json.loads(line));sent+=len(line.encode())
    time.sleep(2)
  report.update(learning_report(out,child.returncode))
 except BaseException as e:
  import traceback
  report['failure_type']=type(e).__name__;report['failure_frames']=[dict(file=Path(x.filename).name,function=x.name,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)][-8:]
 finally:
  if child is not None and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=5)
  report['elapsed_seconds']=round(time.monotonic()-started,2);(out/'report.json').write_text(json.dumps(report,indent=2))
  log=out/'native.log'
  if log.is_file():
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-20000));tail=f.read().decode('utf-8',errors='replace')
   (out/'native-log-tail.json').write_text(json.dumps(dict(tail=tail,private_diagnostic_only=True)))
  base.seal(out,work,report);emit('worker_finished',**report,remote_result_retrieval_verified=False);signal.setitimer(signal.ITIMER_REAL,0)
 return 0 if report['worker_passed'] else 1
if __name__=='__main__':sys.exit(main())