"""One authorized <=600s worker including input setup, native resume, sealing/upload."""
import argparse,json,os,re,signal,sys,time,traceback,urllib.request,zipfile,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flux.distill_resume_contract import *
from flux.run_smoke import launcher,upload,decode_control
from flux.live_monitor import run_live,offline_environment
from flux.distill_diagnostics import monitored_download
def emit(event,**kw):print(json.dumps({'event':event,**kw}),flush=True)
def main():
 started=time.monotonic();p=argparse.ArgumentParser()
 for n in ('task-id','expected-commit','upload-url','probe-url','upload-headers-b64','provenance-b64','teachers-b64','datasets-b64','student-b64','scope-b64'):p.add_argument('--'+n,required=True)
 a=p.parse_args();assert re.fullmatch(r'TASK_\d{8}_\d+',a.task_id)
 scope=validate_scope(decode_control(a.scope_b64));assert scope['code_commit']==a.expected_commit
 headers=decode_control(a.upload_headers_b64);provenance=decode_control(a.provenance_b64)
 assert set(provenance).issubset({'thread_id','turn_id'})
 teachers=decode_control(a.teachers_b64)['teachers'];datasets=decode_control(a.datasets_b64)['datasets'];student=decode_control(a.student_b64)
 assert len(teachers)==5 and len(datasets)==4
 for c,r in zip(teachers,TEACHERS):assert set(c)=={'url','bytes','sha256'} and (c['bytes'],c['sha256'])==(r[3],r[4])
 for c,r in zip(datasets,DATA):assert set(c)=={'url','bytes','sha256'} and (c['bytes'],c['sha256'])==(r[2],r[3])
 assert set(student)=={'url','bytes','sha256'} and (student['bytes'],student['sha256'])==(SOURCE_BYTES,SOURCE_SHA)
 out=ROOT/'outputs'/('five-resume-'+a.task_id);out.mkdir(parents=True,exist_ok=False)
 def hard_exit():time.sleep(max(0,598-(time.monotonic()-started)));os._exit(124)
 threading.Thread(target=hard_exit,daemon=True).start()
 def deadline(*args):raise TimeoutError('600s continuation deadline')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(550)
 passed=False;stage='preflight';env=offline_environment(os.environ)
 env.update(CUDA_VISIBLE_DEVICES='0',PHP_G1_TASK_ID=a.task_id,PHP_DISTILL_SOFT_DEADLINE=str(started+SOFT_SECONDS))
 def run(cmd,seconds,name):
  left=min(seconds,TRAIN_HARD_SECONDS-(time.monotonic()-started))
  if left<=0:raise TimeoutError('No remaining continuation time')
  return run_live(cmd,env,left,out/name,ROOT)
 emit('five_distill_worker_begin',worker_seconds_max=600,trial_count=1,automatic_retries=0,num_envs=80,updates_max=400,source_task='TASK_20261027_216')
 try:
  from flux.preflight import static_checks
  static_checks(expected_commit=a.expected_commit)
  probe=out/'probe.json';probe.write_text(json.dumps({'task_id':a.task_id,'resume':True,'worker_seconds_max':600}))
  upload(a.probe_url,headers,probe);emit('five_distill_delivery_probe_passed')
  python=launcher();stage='dependencies';run([str(python),str(ROOT/'flux/runtime_versions.py')],30,'runtime.log')
  private=out/'upstream-wheels';private.mkdir();wheels=json.loads((ROOT/'flux/smoke-upstream-wheels.json').read_text());assert len(wheels)==8
  def download_wheel(r):
   assert r['url'].startswith('https://files.pythonhosted.org/packages/') and Path(r['file']).name==r['file']
   path=private/r['file'];left=r['bytes']
   with urllib.request.urlopen(r['url'],timeout=30) as source,path.open('xb') as target:
    while left:
     b=source.read(min(left,1048576))
     if not b:raise ValueError('Truncated wheel')
     target.write(b);left-=len(b)
    if source.read(1):raise ValueError('Oversized wheel')
   assert digest(path)==r['sha256']
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download_wheel,wheels))
  inputs=out/'inputs'
  run([str(python),str(ROOT/'flux/prepare_inputs.py'),'--output',str(inputs),'--install','--python',str(python),'--withheld-wheel-dir',str(private)],250,'dependencies.log')
  env['PYTHONPATH']=os.pathsep.join([str(inputs/'.flux-deps'),str(ROOT),str(ROOT/'thirdparty/holosoma/src/holosoma'),env.get('PYTHONPATH','')])
  stage='pinned_inputs';source_dir=out/'private-inputs';source_dir.mkdir();emit('five_distill_stage_begin',stage=stage)
  from flux.parallel_download import download
  jobs=[('teacher-'+str(i)+'.pt',c) for i,c in enumerate(teachers)]+[('dataset-'+str(i)+'.tar.gz',c) for i,c in enumerate(datasets)]+[('source-student.pt',student)]
  def get(row):return monitored_download(row[0],row[1],source_dir,out/'input-diagnostics',download,emit)
  with ThreadPoolExecutor(max_workers=3) as pool:paths=list(pool.map(get,jobs))
  components=json.loads((ROOT/'release-assets.json').read_text())['components'];data_paths=[inputs/'motions-locomotion']
  for i,path in enumerate(paths[5:9]):data_paths.append(unpack_data(path,source_dir/('dataset-'+str(i)),components[COMPONENTS[i+1]]))
  emit('five_distill_inputs_verified',teachers=5,source_student=True,motion_counts=[100,56,60,40,40])
  if time.monotonic()-started>360:raise TimeoutError('Insufficient simulator time inside 600s total budget')
  train_out=out/'training';plan=out/'plan.json'
  plan.write_text(json.dumps({'scope':scope,'provenance':provenance,'task_id':a.task_id,'teachers':[str(x) for x in paths[:5]],
   'datasets':list(map(str,data_paths)),'student':str(paths[-1]),'output':str(train_out)},indent=2))
  stage='distillation';run([str(python),str(ROOT/'flux/five_distill_resume_student.py'),'--plan',str(plan)],540,'distillation.log')
  result=json.loads((train_out/'distillation-result.json').read_text())
  assert result['pipeline_passed'] and result['teachers_frozen_verified'] and result['native_reload_verified'] and result['resume_policy_optimizer_verified']
  assert 0<result['completed_updates']<=400
  passed=True;emit('five_distill_worker_passed',completed_updates=result['completed_updates'])
 except BaseException as exc:
  failure={'stage':stage,'exception_type':type(exc).__name__,'elapsed_seconds':round(time.monotonic()-started,2),
   'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]}
  (out/'failure.json').write_text(json.dumps(failure,indent=2));emit('five_distill_worker_failed',**failure)
 finally:
  signal.alarm(max(1,int(595-(time.monotonic()-started))))
  records=[];final_exists=any((out/'training').rglob('student_final.pt')) if (out/'training').exists() else False
  for path in sorted(out.rglob('*')):
   rel=path.relative_to(out)
   if not path.is_file() or path.is_symlink() or {'private-inputs','inputs','upstream-wheels'}.intersection(rel.parts):continue
   if path.suffix=='.pt' and path.name!=('student_final.pt' if final_exists else 'student_shakeout.pt'):continue
   if path.suffix not in ('.json','.jsonl','.yaml','.yml','.log','.pt','.wandb'):continue
   records.append({'path':str(rel),'bytes':path.stat().st_size,'sha256':digest(path)})
  manifest=out/'result-manifest.json';manifest.write_text(json.dumps({'worker_passed':passed,'files':records},indent=2));bundle=out/'five-distill-result.zip'
  with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   z.write(manifest,manifest.name)
   for r in records:z.write(out/r['path'],r['path'])
  try:upload(a.upload_url,headers,bundle);emit('five_distill_result_uploaded',bytes=bundle.stat().st_size,sha256=digest(bundle))
  except BaseException as exc:passed=False;emit('five_distill_upload_failed',exception_type=type(exc).__name__)
  signal.alarm(0)
 emit('five_distill_entrypoint_complete',passed=passed,elapsed_seconds=round(time.monotonic()-started,2))
 return 0 if passed else 1
if __name__=='__main__':sys.exit(main())