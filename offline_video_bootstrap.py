"""Single bounded 900-second inference worker. Account secrets never reach child."""
import argparse,hashlib,json,os,signal,subprocess,sys,threading,time,zipfile
from pathlib import Path
from scoped_transport import digest,deadline,transfer,validate_capability
from range_input import download
from checkpoint_source import validate_flux_checkpoint
from local_trial_inputs import materialize
from bounded_training_trial import unpack
from inline_execution import decode_inline

SCOPE={'max_tasks':1,'gpu_num':1,'worker_seconds_max':900,'automatic_retries':0,
       'training_updates':0,'preview_seconds':20,'max_policy_steps':2000,'num_envs':10}

def validate_task_binding(config,task_id):
    # gm-run consumes --task_id for its SDK; it does not guarantee a
    # PHP_FLUX_TASK_ID environment variable for the Python entrypoint.
    if not isinstance(task_id,str) or not task_id.startswith('TASK_'):raise ValueError('Explicit task identity required')
    if config.get('scope')!=SCOPE or config.get('task_id')!=task_id:raise ValueError('Task scope binding')

def main():
    started=time.monotonic();active=[None]
    def guard():
        time.sleep(max(0,started+900-time.monotonic()))
        if active[0] is not None and active[0].poll() is None:
            try:os.killpg(active[0].pid,signal.SIGKILL)
            except ProcessLookupError:pass
        os._exit(124)
    threading.Thread(target=guard,daemon=True).start()
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--task-id',required=True)
    a=p.parse_args();raw,_=decode_inline(a.config,a.sha256);c=json.loads(raw)
    root=Path(__file__).resolve().parent
    validate_task_binding(c,a.task_id)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()!=c['code_commit']:raise ValueError('Code pin')
    validate_capability(c['output_cap'],'PUT');validate_flux_checkpoint(c['checkpoint']['cap'],'GET')
    work=root/'private-video';work.mkdir(exist_ok=False);out=work/'output';out.mkdir()
    state={'scope':SCOPE,'task_id':c['task_id'],'code_commit':c['code_commit'],'_wb_agent':c['_wb_agent'],
           'worker_passed':False,'stage':'inputs','native_returncode':None}
    def stage(name):state['stage']=name;print(json.dumps({'event':'video_stage','stage':name}),flush=True)
    try:
        materialize(root/'inputs',work,manifest_sha256=c['shard_manifest_sha256'],
                    absolute_deadline=started+180,unpack_archive=unpack)
        payload=work/'payload';bundle=payload/'bundle';checkpoint=work/'source-final.pt'
        stage('checkpoint')
        download(c['checkpoint']['cap'],checkpoint,size=c['checkpoint']['bytes'],sha256=c['checkpoint']['sha256'],
                 absolute_deadline=started+240,cap_validator=validate_flux_checkpoint)
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        stage('dependencies')
        with (out/'install.log').open('xb') as log:
            child=subprocess.Popen(['/isaac-sim/python.sh','-m','pip','install','--isolated','--disable-pip-version-check',
                '--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes',
                '--find-links',str(payload/'deps/wheels'),'--target',str(payload/'dependency-site'),'-r',str(payload/'deps/overlay.lock')],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            active[0]=child
            try:code=child.wait(timeout=max(.1,started+300-time.monotonic()))
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
                active[0]=None
            if code:raise RuntimeError('Dependency install failed')
        req={k:c[k] for k in ['scope','task_id','_wb_agent','source_plan','source_completed_updates']}
        req.update(output=str(out),bundle=str(bundle),data_root=str(payload/'formal_data'),
            teachers=[str(payload/'teachers'/f'{i}.pt') for i in range(5)],checkpoint=str(checkpoint),source_checkpoint_sha256=c['checkpoint']['sha256'])
        req_path=work/'video-request.json';req_path.write_text(json.dumps(req))
        plan_path=work/'source-plan.json';plan_path.write_text(json.dumps(c['source_plan']))
        sys.path.insert(0,str(payload))
        from student_adapter.worker_budget import isolated_child_env
        # Use the established child environment allowlist: no credentials or transfer URLs.
        paths=[str(root),str(payload),str(payload/'dependency-site'),str(bundle/'overlay'),str(bundle/'runtime/php'),
               str(bundle/'runtime/holosoma/src/holosoma'),str(bundle),str(payload/'native-source')]
        child_env=isolated_child_env(os.environ,pythonpath=os.pathsep.join(paths),output=out,
            asset_root=bundle/'asset',train_deadline=started+720,task_id=c['task_id'])
        child_env.update(HUD04_VIDEO_REQUEST=str(req_path),HUD04_FORMAL_PLAN=str(plan_path),HUD04_VIDEO_DEADLINE=str(started+720))
        for key in ['ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT']:
            if key in os.environ:child_env[key]=os.environ[key]
        stage('student_preview')
        with (out/'native.log').open('xb') as log:
            child=subprocess.Popen(['/isaac-sim/python.sh',str(root/'offline_video_child.py')],cwd=payload,env=child_env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[0]=child
            try:state['native_returncode']=child.wait(timeout=max(.1,started+780-time.monotonic()))
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
                active[0]=None
        state['video_evidence_present']=(out/'video-evidence.json').is_file()
        state['worker_passed']=state['native_returncode']==0 and state['video_evidence_present']
        stage('complete' if state['worker_passed'] else 'native_failure')
    except Exception as e:
        state['error_type']=type(e).__name__
        print(json.dumps({'event':'video_failure','stage':state['stage'],'error_type':type(e).__name__}),flush=True)
    state['elapsed_seconds']=time.monotonic()-started
    (out/'bootstrap-result.json').write_text(json.dumps(state,indent=2))
    with deadline(started+820):
        files=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(out.rglob('*')) if f.is_file()]
        if sum(f['bytes'] for f in files)>450*1024**2:raise ValueError('Result too large')
        (out/'result-manifest.json').write_text(json.dumps({'scope':SCOPE,'files':files},indent=2))
        archive=work/'video-result.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(out))
    receipt=transfer(c['output_cap'],archive,method='PUT',size=archive.stat().st_size,sha256=digest(archive),absolute_deadline=started+890)
    print(json.dumps({'event':'video_result_uploaded',**receipt,'worker_passed':state['worker_passed']}),flush=True)
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print(json.dumps({'event':'video_bootstrap_failed','error_type':type(e).__name__}),flush=True);sys.exit(1)
