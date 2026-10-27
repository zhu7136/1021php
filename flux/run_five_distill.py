"""ONE authorized worker <=1800s including setup, pilot, sealing and upload."""
import argparse,json,os,re,signal,sys,time,traceback,urllib.request,zipfile,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flux.five_distill_contract import *
from flux.run_smoke import launcher,upload,decode_control
from flux.live_monitor import run_live,offline_environment
from flux.distill_diagnostics import monitored_download

def emit(event,**kw):print(json.dumps({'event':event,**kw}),flush=True)

def main():
 started=time.monotonic();p=argparse.ArgumentParser()
 for n in ('task-id','expected-commit','upload-url','probe-url','upload-headers-b64','provenance-b64','teachers-b64','datasets-b64','scope-b64'):p.add_argument('--'+n,required=True)
 a=p.parse_args()
 if not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id):raise ValueError('Invalid task ID')
 scope=validate_scope(decode_control(a.scope_b64));assert scope['code_commit']==a.expected_commit
 headers=decode_control(a.upload_headers_b64);controls={**decode_control(a.teachers_b64),**decode_control(a.datasets_b64)};provenance=decode_control(a.provenance_b64)
 if not set(provenance).issubset({'thread_id','turn_id'}):raise ValueError('Invalid provenance')
 if set(controls)!={'teachers','datasets'} or len(controls['teachers'])!=5 or len(controls['datasets'])!=4:raise ValueError('Input count changed')
 for c,r in zip(controls['teachers'],TEACHERS):
  if set(c)!={'url','bytes','sha256'} or (c['bytes'],c['sha256'])!=(r[3],r[4]):raise ValueError('Wrong pinned teacher')
 for c,r in zip(controls['datasets'],DATA):
  if set(c)!={'url','bytes','sha256'} or (c['bytes'],c['sha256'])!=(r[2],r[3]):raise ValueError('Wrong pinned dataset')
 out=ROOT/'outputs'/('five-distill-'+a.task_id);out.mkdir(parents=True,exist_ok=False)
 # Last-resort process deadline also covers Python executor waits and sealing.
 # The simulator child has its earlier 1580s group-killing supervisor deadline.
 def hard_exit():
  time.sleep(max(0,1798-(time.monotonic()-started)));os._exit(124)
 threading.Thread(target=hard_exit,daemon=True).start()
 def deadline(*args):raise TimeoutError('Entire distillation worker deadline')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(1600)
 passed=False;stage='preflight';env=offline_environment(os.environ)
 env.update(CUDA_VISIBLE_DEVICES='0',PHP_G1_TASK_ID=a.task_id,PHP_DISTILL_SOFT_DEADLINE=str(started+SOFT_SECONDS))
 def run(cmd,seconds,name):
  left=min(seconds,TRAIN_HARD_SECONDS-(time.monotonic()-started))
  if left<=0:raise TimeoutError('No remaining worker time')
  return run_live(cmd,env,left,out/name,ROOT)
 emit('five_distill_worker_begin',worker_seconds_max=1800,trial_count=1,automatic_retries=0,skills=list(SKILLS),num_envs=80)
 try:
  from flux.preflight import static_checks
  static_checks(expected_commit=a.expected_commit)
  probe=out/'probe.json';probe.write_text(json.dumps({'task_id':a.task_id,'mode':'masked_pure_dagger'}))
  upload(a.probe_url,headers,probe);emit('five_distill_delivery_probe_passed')
  python=launcher();stage='dependencies'
  run([str(python),str(ROOT/'flux/runtime_versions.py')],30,'runtime.log')
  private=out/'upstream-wheels';private.mkdir();wheels=json.loads((ROOT/'flux/smoke-upstream-wheels.json').read_text())
  assert len(wheels)==8
  def download_wheel(r):
   if not r['url'].startswith('https://files.pythonhosted.org/packages/') or Path(r['file']).name!=r['file']:raise ValueError('Wrong wheel source')
   path=private/r['file'];left=r['bytes']
   with urllib.request.urlopen(r['url'],timeout=30) as source,path.open('xb') as target:
    while left:
     b=source.read(min(left,1048576))
     if not b:raise ValueError('Truncated wheel')
     target.write(b);left-=len(b)
    if source.read(1):raise ValueError('Oversized wheel')
   if digest(path)!=r['sha256']:raise ValueError('Wheel hash mismatch')
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download_wheel,wheels))
  inputs=out/'inputs'
  run([str(python),str(ROOT/'flux/prepare_inputs.py'),'--output',str(inputs),'--install','--python',str(python),'--withheld-wheel-dir',str(private)],250,'dependencies.log')
  env['PYTHONPATH']=os.pathsep.join([str(inputs/'.flux-deps'),str(ROOT),str(ROOT/'thirdparty/holosoma/src/holosoma'),env.get('PYTHONPATH','')])
  stage='pinned_inputs';source_dir=out/'private-inputs';source_dir.mkdir()
  emit('five_distill_stage_begin',stage=stage)
  from flux.parallel_download import download
  jobs=[('teacher-'+str(i)+'.pt',c) for i,c in enumerate(controls['teachers'])]+[('dataset-'+str(i)+'.tar.gz',c) for i,c in enumerate(controls['datasets'])]
  def get(row):
   name,c=row;return monitored_download(name,c,source_dir,out/'input-diagnostics',download,emit)
  with ThreadPoolExecutor(max_workers=3) as pool:paths=list(pool.map(get,jobs))
  components=json.loads((ROOT/'release-assets.json').read_text())['components']
  datasets=[inputs/'motions-locomotion']
  for i,path in enumerate(paths[5:]):datasets.append(unpack_data(path,source_dir/('dataset-'+str(i)),components[COMPONENTS[i+1]]))
  emit('five_distill_inputs_verified',teachers=5,motion_counts=[100,56,60,40,40])
  if time.monotonic()-started>1050:raise TimeoutError('Insufficient pilot time after setup')
  train_out=out/'training'
  plan=out/'plan.json';plan.write_text(json.dumps({'scope':scope,'provenance':provenance,'task_id':a.task_id,
   'teachers':[str(x) for x in paths[:5]],'datasets':list(map(str,datasets)),'output':str(train_out)},indent=2))
  stage='distillation'
  run([str(python),str(ROOT/'flux/five_distill_student.py'),'--plan',str(plan)],1580,'distillation.log')
  result=json.loads((train_out/'distillation-result.json').read_text())
  if not result.get('pipeline_passed') or not result.get('teachers_frozen_verified') or not result.get('native_reload_verified'):raise ValueError('Native result not verified')
  passed=True;emit('five_distill_worker_passed',completed_updates=result['completed_updates'])
 except BaseException as exc:
  failure={'stage':stage,'exception_type':type(exc).__name__,'elapsed_seconds':round(time.monotonic()-started,2),
   'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]}
  (out/'failure.json').write_text(json.dumps(failure,indent=2));emit('five_distill_worker_failed',**failure)
 finally:
  signal.alarm(max(1,int(1795-(time.monotonic()-started))))
  records=[];final_exists=any((out/'training').rglob('student_final.pt')) if (out/'training').exists() else False
  for path in sorted(out.rglob('*')):
   rel=path.relative_to(out)
   if not path.is_file() or path.is_symlink() or {'private-inputs','inputs','upstream-wheels'}.intersection(rel.parts):continue
   if path.suffix=='.pt' and path.name!=('student_final.pt' if final_exists else 'student_shakeout.pt'):continue
   if path.suffix not in ('.json','.jsonl','.yaml','.yml','.log','.pt','.wandb'):continue
   records.append({'path':str(rel),'bytes':path.stat().st_size,'sha256':digest(path)})
  manifest=out/'result-manifest.json';manifest.write_text(json.dumps({'worker_passed':passed,'files':records},indent=2))
  bundle=out/'five-distill-result.zip'
  with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   z.write(manifest,manifest.name)
   for r in records:z.write(out/r['path'],r['path'])
  try:upload(a.upload_url,headers,bundle);emit('five_distill_result_uploaded',bytes=bundle.stat().st_size,sha256=digest(bundle))
  except BaseException as exc:passed=False;emit('five_distill_upload_failed',exception_type=type(exc).__name__)
  signal.alarm(0)
 emit('five_distill_entrypoint_complete',passed=passed,elapsed_seconds=round(time.monotonic()-started,2))
 return 0 if passed else 1

if __name__=='__main__':sys.exit(main())