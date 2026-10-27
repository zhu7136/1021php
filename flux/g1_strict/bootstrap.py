"""One G1 strict-close control:2env/reset+8zero steps/0PPO/900s/30sclose/retry0."""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
from dependency_install import workflow,command
from fetch_official_wheels import absolute_budget
from probe_common import environment

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_sources(root):
 spec=json.loads((root/'source-manifest.json').read_text())
 for row in spec['files']:
  p=root/row['path']
  if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('source manifest mismatch')
 return sha(root/'source-manifest.json')
def collect(output):
 result={}
 for name in ['probe.json','runtime.json','helper-before.json','helper-at-close.json','close-settings.json','close-returned.json','lifecycle-returned.json','child-error.json']:
  p=output/name
  if p.exists():
   if p.stat().st_size>30000:raise ValueError('oversized child receipt')
   result[name]=json.loads(p.read_text())
 trace=output/'close-stack.log'
 if trace.exists():result['close_stack_tail']=trace.read_text(errors='replace')[-12000:]
 return result
def stop(process):
 if process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=2)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
def run_child(argv,output,env,deadline,emit):
 log=output/'child.log';rows=[];seen=0;close_started=None;reason='exited';process=None
 with log.open('xb') as stream:
  process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   while True:
    journal=output/'events.jsonl'
    if journal.exists():
     lines=journal.read_text().splitlines()
     for line in lines[seen:]:
      try:row=json.loads(line)
      except ValueError:break
      seen+=1;rows.append(row);emit('g1_strict_stage',**row)
      if row['stage']=='app_close_started':close_started=row['monotonic']
    code=process.poll()
    if code is not None:break
    now=time.monotonic()
    if close_started is not None and now-close_started>30:reason='close_deadline';stop(process);break
    if now>deadline:reason='worker_deadline';stop(process);break
    time.sleep(.2)
  finally:stop(process)
 raw=log.read_bytes();receipts=collect(output)
 passed=process.returncode==0 and reason=='exited' and receipts.get('close-returned.json',{}).get('returned') is True and receipts.get('lifecycle-returned.json',{}).get('returned') is True and receipts.get('probe.json',{}).get('explicit_control_steps')==8 and receipts.get('probe.json',{}).get('finite_observations_and_rewards') is True and receipts.get('close-settings.json',{}).get('readback') is False and receipts.get('helper-at-close.json',{}).get('usd_method_unchanged') is True
 return dict(child_exit_code=process.returncode,supervisor_reason=reason,strict_close_passed=passed,events=rows,receipts=receipts,child_log_bytes=len(raw),child_log_sha256=hashlib.sha256(raw).hexdigest(),child_log_tail=raw[-7000:].decode(errors='replace') if not passed else '')
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--task-id');p.add_argument('--commit');p.add_argument('--thread-id');p.add_argument('--turn-id');a=p.parse_args()
 if not a.execute:
  print(json.dumps(dict(execution_requested=False,worker_seconds_max=900,close_seconds_max=30,num_envs=2,explicit_control_steps=8,ppo_updates=0,automatic_retries=0)),flush=True);return 0
 if not a.task_id or not re.fullmatch('TASK_[0-9]{8}_[0-9]+',a.task_id) or not a.commit or not re.fullmatch('[0-9a-f]{40}',a.commit) or not a.thread_id or not a.turn_id:raise ValueError('missing task/commit/provenance')
 started=time.monotonic();deadline=started+900;root=Path(__file__).resolve().parent;repo=root.parents[1]
 def emit(event,**kw):print(json.dumps(dict(event=event,task_id=a.task_id,source_commit=a.commit,**kw),separators=(',',':')),flush=True)
 state=dict(passed=False,num_envs=2,explicit_control_steps=8,ppo_updates=0,automatic_retries=0,worker_seconds_max=900,close_seconds_max=30,robot='g1',native_close_root_cause_identified=False)
 try:
  with absolute_budget(max(.001,deadline-time.monotonic()-5)):
   if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,env=environment(),text=True,timeout=10).strip()!=a.commit:raise ValueError('source checkout mismatch')
   state['source_manifest_sha256']=verify_sources(root)
   sys.path.insert(0,str(repo))
   from flux.preflight import static_checks
   state['source_preflight']=static_checks(expected_commit=a.commit);emit('g1_strict_source_verified',**state['source_preflight'])
   installed=workflow(root,a.task_id,deadline,emit);state['dependency_receipt']=installed
   out=root/('probe-'+a.task_id);out.mkdir(exist_ok=False)
   inputs=out/'inputs'
   prepared=command(['/isaac-sim/python.sh',str(repo/'flux/prepare_inputs.py'),'--output',str(inputs)],out/'prepare.log',deadline-180,160)
   emit('g1_strict_inputs_prepared',**prepared)
   if prepared['returncode']!=0:raise ValueError('pinned public G1 input reconstruction failed')
   env=environment();site=root/('isolation-'+a.task_id)/'site'
   env['PYTHONPATH']=os.pathsep.join([str(site),str(root),str(repo),str(repo/'thirdparty/holosoma/src/holosoma'),env.get('PYTHONPATH','')])
   if deadline-time.monotonic()<120:raise TimeoutError('insufficient simulator and cleanup budget')
   emit('g1_strict_child_launch',remaining_seconds=round(deadline-time.monotonic(),3),ppo_updates=0)
   state.update(run_child(['/isaac-sim/python.sh',str(root/'child.py'),'--repo',str(repo),'--data',str(inputs/'motions-locomotion'),'--output',str(out),'--thread-id',a.thread_id,'--turn-id',a.turn_id],out,env,deadline-12,emit))
   state['passed']=state['strict_close_passed']
 except Exception as error:
  state.update(error_type=type(error).__name__,error_message=str(error)[:500],frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-8:]])
 state['elapsed_seconds']=round(time.monotonic()-started,3);emit('g1_strict_final',**state);return 0 if state['passed'] else 1
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as error:print(json.dumps(dict(event='g1_strict_entry_failure',error_type=type(error).__name__,passed=False)),flush=True);sys.exit(1)