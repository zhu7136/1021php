"""Single <=600s GPU worker incl setup, frozen inference, and output delivery."""
import argparse,json,os,re,signal,sys,time,traceback,urllib.request,zipfile,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flux.student_video_contract import *
from flux.run_smoke import launcher,upload,decode_control
from flux.live_monitor import run_live,offline_environment
from flux.distill_diagnostics import monitored_download
def emit(event,**kw):print(json.dumps({'event':event,**kw}),flush=True)
def main():
 started=time.monotonic();p=argparse.ArgumentParser()
 for n in ('task-id','expected-commit','upload-url','probe-url','upload-headers-b64','provenance-b64','datasets-b64','student-b64','scope-b64'):p.add_argument('--'+n,required=True)
 a=p.parse_args();assert re.fullmatch(r'TASK_\d{8}_\d+',a.task_id)
 scope=validate_scope(decode_control(a.scope_b64));assert scope['code_commit']==a.expected_commit
 headers=decode_control(a.upload_headers_b64);provenance=decode_control(a.provenance_b64)
 assert set(provenance).issubset({'thread_id','turn_id'})
 datasets=decode_control(a.datasets_b64)['datasets'];student=decode_control(a.student_b64);assert len(datasets)==4
 for c,r in zip(datasets,DATA):assert set(c)=={'url','bytes','sha256'} and (c['bytes'],c['sha256'])==(r[2],r[3])
 assert set(student)=={'url','bytes','sha256'} and (student['bytes'],student['sha256'])==(SOURCE_BYTES,SOURCE_SHA)
 out=ROOT/'outputs'/('student-videos-'+a.task_id);out.mkdir(parents=True,exist_ok=False)
 def hard_exit():time.sleep(max(0,598-(time.monotonic()-started)));os._exit(124)
 threading.Thread(target=hard_exit,daemon=True).start()
 def deadline(*args):raise TimeoutError('600s worker deadline')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(550)
 passed=False;stage='preflight';env=offline_environment(os.environ);env['CUDA_VISIBLE_DEVICES']='0'
 def run(cmd,seconds,name):
  left=min(seconds,540-(time.monotonic()-started))
  if left<=0:raise TimeoutError('No remaining replay time')
  return run_live(cmd,env,left,out/name,ROOT)
 emit('student_video_worker_begin',worker_seconds_max=600,training_updates=0,trial_count=1,automatic_retries=0)
 try:
  from flux.preflight import static_checks
  static_checks(expected_commit=a.expected_commit)
  probe=out/'probe.json';probe.write_text(json.dumps({'task_id':a.task_id,'evaluation_only':True,'worker_seconds_max':600}))
  upload(a.probe_url,headers,probe);emit('student_video_delivery_probe_passed')
  python=launcher();stage='dependencies';run([str(python),str(ROOT/'flux/runtime_versions.py')],30,'runtime.log')
  private=out/'upstream-wheels';private.mkdir();wheels=json.loads((ROOT/'flux/smoke-upstream-wheels.json').read_text())
  encoder=json.loads((ROOT/'flux/eval-encoder-wheel.json').read_text());wheels.append(encoder)
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
  run([str(python),str(ROOT/'flux/prepare_inputs.py'),'--output',str(inputs),'--install','--python',str(python),'--withheld-wheel-dir',str(private)],240,'dependencies.log')
  run([str(python),'-m','pip','install','--no-deps','--no-index','--target',str(inputs/'.flux-deps'),str(private/encoder['file'])],30,'encoder-install.log')
  env['PYTHONPATH']=os.pathsep.join([str(inputs/'.flux-deps'),str(ROOT),str(ROOT/'thirdparty/holosoma/src/holosoma'),env.get('PYTHONPATH','')])
  stage='pinned_inputs';source_dir=out/'private-inputs';source_dir.mkdir()
  from flux.parallel_download import download
  jobs=[('dataset-'+str(i)+'.tar.gz',c) for i,c in enumerate(datasets)]+[('source-student.pt',student)]
  def get(row):return monitored_download(row[0],row[1],source_dir,out/'input-diagnostics',download,emit)
  with ThreadPoolExecutor(max_workers=3) as pool:paths=list(pool.map(get,jobs))
  components=json.loads((ROOT/'release-assets.json').read_text())['components'];data_paths=[inputs/'motions-locomotion']
  for i,path in enumerate(paths[:4]):data_paths.append(unpack_data(path,source_dir/('dataset-'+str(i)),components[COMPONENTS[i+1]]))
  emit('student_video_inputs_verified',source_student=True,motion_counts=list(COUNTS),teacher_downloads=0)
  if time.monotonic()-started>400:raise TimeoutError('Insufficient simulator time inside 600s total budget')
  plan=out/'plan.json';plan.write_text(json.dumps({'scope':scope,'provenance':provenance,'task_id':a.task_id,
   'datasets':list(map(str,data_paths)),'student':str(paths[-1]),'output':str(out/'replay'),'soft_deadline':started+520},indent=2))
  stage='frozen_student_replay';run([str(python),str(ROOT/'flux/student_video_inference.py'),'--plan',str(plan)],540,'replay.log')
  result=json.loads((out/'replay/replay-result.json').read_text())
  assert result['pipeline_passed'] and result['training_updates']==0 and result['strict_model_unchanged']
  passed=True;emit('student_video_worker_passed',clips=len(result['clips']))
 except BaseException as exc:
  failure={'stage':stage,'exception_type':type(exc).__name__,'elapsed_seconds':round(time.monotonic()-started,2),
   'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]}
  (out/'failure.json').write_text(json.dumps(failure,indent=2));emit('student_video_worker_failed',**failure)
 finally:
  signal.alarm(max(1,int(595-(time.monotonic()-started))))
  records=[]
  for path in sorted(out.rglob('*')):
   rel=path.relative_to(out)
   if not path.is_file() or path.is_symlink() or {'private-inputs','inputs','upstream-wheels'}.intersection(rel.parts):continue
   if path.suffix not in ('.json','.jsonl','.yaml','.yml','.log','.mp4'):continue
   records.append({'path':str(rel),'bytes':path.stat().st_size,'sha256':digest(path)})
  manifest=out/'result-manifest.json';manifest.write_text(json.dumps({'worker_passed':passed,'training_updates':0,'source_sha256':SOURCE_SHA,'files':records},indent=2))
  bundle=out/'student-videos-result.zip'
  with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   z.write(manifest,manifest.name)
   for r in records:z.write(out/r['path'],r['path'])
  try:upload(a.upload_url,headers,bundle);emit('student_video_result_uploaded',bytes=bundle.stat().st_size,sha256=digest(bundle))
  except BaseException as exc:passed=False;emit('student_video_upload_failed',exception_type=type(exc).__name__)
  signal.alarm(0)
 emit('student_video_entrypoint_complete',passed=passed,elapsed_seconds=round(time.monotonic()-started,2))
 return 0 if passed else 1
if __name__=='__main__':sys.exit(main())