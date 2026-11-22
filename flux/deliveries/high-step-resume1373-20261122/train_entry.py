"""Pinned offline Git-shard delivery; includes reassembly in worker time bound."""
import argparse,importlib.util,json,shutil,signal,subprocess,sys,time
from pathlib import Path
import local_shards
PIN='581b77fa2188559d8fd15c8e39af2db025762b7418846e41212cd4a45a0221d0'
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--execute',action='store_true');p.add_argument('--task-id');p.add_argument('--commit');p.add_argument('--work',type=Path)
 a=p.parse_args();started=time.monotonic()
 if a.execute:
  if not a.commit:raise ValueError('commit_required')
  actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
  if actual!=a.commit:raise ValueError('git_commit_mismatch')
  def deadline(*args):raise TimeoutError('worker_1800s_budget')
  signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,1800)
 dest=a.work or ROOT/('delivery-'+(a.task_id or 'cpu-check'))
 receipt=local_shards.reassemble(ROOT/'inputs',dest,manifest_sha256=PIN)
 for name in ['overrides']:
  shutil.copytree(ROOT/name,dest/name)
 for name in ['PLAN.json','train_high_step.py']:
  shutil.copyfile(ROOT/name,dest/name)
 print(json.dumps(dict(event='git_shards_reassembled',manifest_sha256=PIN,files=len(receipt['files']),source_iteration=1373)),flush=True)
 spec=importlib.util.spec_from_file_location('high_step_worker',dest/'train_high_step.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
 worker.OUTER_STARTED=started
 sys.argv=[str(dest/'train_high_step.py'),'--work',str(dest/'work')]+(['--execute','--task-id',a.task_id] if a.execute else [])
 return worker.main()
if __name__=='__main__':sys.exit(main())
