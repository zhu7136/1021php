"""Exactly one frozen eval; entire worker <=1800s including setup/sealing/upload."""
import argparse,json,os,signal,sys,time,traceback,subprocess,urllib.request,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flux.run_smoke import launcher,upload,decode_control
from flux.g1_eval283_protocol import *
from flux.live_monitor import run_live,offline_environment

def emit(event,**kw):print(json.dumps({'event':event,**kw}),flush=True)

def main():
    started=time.monotonic();p=argparse.ArgumentParser()
    for name in ('task-id','expected-commit','upload-url','probe-url','upload-headers-b64',
                 'aria-provenance-b64','lowstep-input-b64','checkpoint-input-b64'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    import re
    if not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id):raise ValueError('Invalid task ID')
    headers=decode_control(a.upload_headers_b64);provenance=decode_control(a.aria_provenance_b64)
    from flux.g1_eval283_input import validate_control,fetch_checkpoint
    source_control=validate_control(decode_control(a.checkpoint_input_b64))
    if not set(provenance).issubset({'thread_id','turn_id'}):raise ValueError('Invalid provenance')
    out=ROOT/'outputs'/('g1-eval283-'+a.task_id);out.mkdir(parents=True,exist_ok=False)
    def deadline(*unused):raise TimeoutError('Entire evaluation worker time limit')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(1680)
    passed=False;stage='source_preflight'
    emit('eval283_worker_begin',worker_seconds_max=1800,training_updates=0,planned_trials=168,automatic_retries=0)
    env=offline_environment(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',PHP_G1_TASK_ID=a.task_id)
    env.pop('PHP_FLUX_RESUME_CHECKPOINT',None)
    def run(cmd,seconds,name):
        remaining=min(seconds,1640-(time.monotonic()-started))
        if remaining<=0:raise TimeoutError('No remaining worker budget')
        return run_live(cmd,env,remaining,out/name,ROOT)
    try:
        from flux.g1_lowstep_input import static_checks,fetch_data
        checks=static_checks(expected_commit=a.expected_commit)
        (out/'source-preflight.json').write_text(json.dumps(checks,indent=2))
        probe=out/'probe.json';probe.write_text(json.dumps({'task_id':a.task_id,'evaluation_only':True,'training_updates':0}))
        upload(a.probe_url,headers,probe);emit('eval283_delivery_probe_passed')
        python=launcher();stage='runtime_and_dependencies'
        run([str(python),str(ROOT/'flux/runtime_versions.py')],30,'runtime.log')
        private=out/'upstream-wheels';private.mkdir()
        wheels=json.loads((ROOT/'flux/smoke-upstream-wheels.json').read_text())
        if len(wheels)!=8:raise ValueError('Unexpected wheel manifest')
        def download(row):
            if not row['url'].startswith('https://files.pythonhosted.org/packages/') or Path(row['file']).name!=row['file']:
                raise ValueError('Invalid dependency source')
            path=private/row['file'];remaining=row['bytes']
            with urllib.request.urlopen(row['url'],timeout=45) as incoming,path.open('xb') as target:
                while remaining:
                    b=incoming.read(min(remaining,1048576))
                    if not b:raise ValueError('Truncated wheel')
                    target.write(b);remaining-=len(b)
                if incoming.read(1):raise ValueError('Oversized wheel')
            if digest(path)!=row['sha256']:raise ValueError('Wheel hash mismatch')
        with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(download,wheels))
        inputs=out/'inputs'
        run([str(python),str(ROOT/'flux/prepare_inputs.py'),'--output',str(inputs),'--install',
             '--python',str(python),'--withheld-wheel-dir',str(private)],240,'dependencies.log')
        env['PYTHONPATH']=os.pathsep.join([str(inputs/'.flux-deps'),str(ROOT),str(ROOT/'thirdparty/holosoma/src/holosoma'),env.get('PYTHONPATH','')])
        stage='encoder_preflight'
        run([str(python),'-m','flux.video_preflight','--output',str(out/'encoder-preflight')],40,'encoder-preflight.log')
        env['PATH']=str(out/'encoder-preflight/bin')+os.pathsep+env.get('PATH','')
        stage='input_verification'
        lowstep=fetch_data(decode_control(a.lowstep_input_b64),out/'lowstep-data')
        source=fetch_checkpoint(source_control,out/'checkpoint-input')
        emit('eval283_inputs_verified',checkpoint_sha256=CHECKPOINT_SHA256,pairs=56)
        scope={'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,'gpu':'1x4090D-24G',
               'worker_seconds_max':1800,'training_updates':0,'source_run':SOURCE_RUN,
               'checkpoint_sha256':CHECKPOINT_SHA256,'dataset':'motions-low-step','pairs':56,
               'terrain_variants':64,'repeats':3,'seeds':list(SEEDS),'output_upload_probe_passed':True,
               'code_commit':a.expected_commit,'aria_provenance':provenance,'task_id':a.task_id}
        validate_scope(scope);approval=out/'approval.json';approval.write_text(json.dumps(scope,indent=2))
        stage='frozen_policy_evaluation'
        if time.monotonic()-started>1200:raise TimeoutError('Insufficient simulator evaluation budget')
        run([str(python),str(ROOT/'flux/g1_eval283_teacher.py'),'--data',str(lowstep),'--checkpoint',str(source),
             '--output',str(out/'evaluation'),'--approval-file',str(approval),
             '--soft-deadline',str(started+1500)],1640,'evaluation.log')
        result=json.loads((out/'evaluation/evaluation_result.json').read_text())
        if not result.get('pipeline_passed') or not result.get('frozen_model_and_optimizer_verified') or result['training_updates']!=0:
            raise ValueError('Frozen evaluation invariants not verified')
        passed=True;emit('eval283_worker_passed',**result)
    except BaseException as exc:
        failure={'stage':stage,'exception_type':type(exc).__name__,'elapsed_seconds':round(time.monotonic()-started,2),
                 'frames':[{'file':Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(exc.__traceback__)]}
        (out/'failure.json').write_text(json.dumps(failure,indent=2));emit('eval283_worker_failed',**failure)
    finally:
        signal.alarm(max(1,int(1795-(time.monotonic()-started))))
        records=[]
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.yaml','.yml','.log','.mp4') and not {'inputs','lowstep-data','checkpoint-input','upstream-wheels'}.intersection(path.relative_to(out).parts):
                records.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':digest(path)})
        manifest=out/'result-manifest.json';manifest.write_text(json.dumps({'worker_passed':passed,'training_updates':0,'files':records},indent=2))
        bundle=out/'g1-evaluation-result.zip'
        with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            z.write(manifest,manifest.name)
            for row in records:z.write(out/row['path'],row['path'])
        try:
            upload(a.upload_url,headers,bundle)
            emit('eval283_result_uploaded',bytes=bundle.stat().st_size,sha256=digest(bundle))
        except BaseException as exc:passed=False;emit('eval283_result_upload_failed',exception_type=type(exc).__name__)
        signal.alarm(0)
    emit('eval283_entrypoint_complete',passed=passed,elapsed_seconds=round(time.monotonic()-started,2))
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())