"""Pinned offline Git-shard delivery; includes reassembly in worker time bound."""
import argparse,importlib.util,json,shutil,signal,subprocess,sys,time
from pathlib import Path
import local_shards
PIN='a261f186c46968d18254931ed31e103aa569c3a81ddfa3765c65eff7413e51d9'
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
 for name in ['PLAN.json','train_locomotion.py']:
  shutil.copyfile(ROOT/name,dest/name)
 print(json.dumps(dict(event='git_shards_reassembled',manifest_sha256=PIN,files=len(receipt['files']),source_iteration=1361)),flush=True)
 spec=importlib.util.spec_from_file_location('locomotion_worker',dest/'train_locomotion.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
 worker.OUTER_STARTED=started
 sys.argv=[str(dest/'train_locomotion.py'),'--work',str(dest/'work')]+(['--execute','--task-id',a.task_id] if a.execute else [])
 return worker.main()
if __name__=='__main__':sys.exit(main())
