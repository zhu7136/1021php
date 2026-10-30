"""One bounded GPU task, only explicitly scoped climbing previews, no retries."""
import argparse,json,os,signal,subprocess,sys,threading,time,zipfile
from pathlib import Path
from scoped_transport import digest,deadline,transfer,validate_capability
from range_input import download
from checkpoint_source import validate_flux_checkpoint
from local_trial_inputs import materialize
from bounded_training_trial import unpack
from inline_execution import decode_inline
from obstacle_video_inputs import SCOPE,isolate

def main():
    started=time.monotonic();active=[None]
    def guard():
        time.sleep(max(0,started+SCOPE['worker_seconds_max']-time.monotonic()))
        if active[0] is not None and active[0].poll() is None:
            try:os.killpg(active[0].pid,signal.SIGKILL)
            except ProcessLookupError:pass
        os._exit(124)
    threading.Thread(target=guard,daemon=True).start()
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--sha256',required=True);p.add_argument('--task-id',required=True)
    a=p.parse_args();raw,_=decode_inline(a.config,a.sha256);c=json.loads(raw)
    if c['scope']!=SCOPE or c['task_id']!=a.task_id or not a.task_id.startswith('TASK_'):raise ValueError('Task/scope identity')
    root=Path(__file__).resolve().parent
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()!=c['code_commit']:raise ValueError('Commit pin')
    validate_capability(c['output_cap'],'PUT');validate_flux_checkpoint(c['checkpoint']['cap'],'GET')
    work=root/'private-obstacle-video';work.mkdir(exist_ok=False);out=work/'output';out.mkdir()
    state={'scope':SCOPE,'task_id':c['task_id'],'code_commit':c['code_commit'],'_wb_agent':c['_wb_agent'],
        'worker_passed':False,'stage':'inputs','cases':[]}
    def stage(name):state['stage']=name;print(json.dumps({'event':'obstacle_video_stage','stage':name}),flush=True)
    def child_run(command,log,env,limit,cwd=None):
        with log.open('xb') as stream:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active[0]=child
            try:return child.wait(timeout=max(.1,limit-time.monotonic()))
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
                active[0]=None
    try:
        materialize(root/'inputs',work,manifest_sha256=c['shard_manifest_sha256'],absolute_deadline=started+min(180,SCOPE['worker_seconds_max']*.35),unpack_archive=unpack)
        payload=work/'payload';bundle=payload/'bundle';checkpoint=work/'source-final.pt'
        stage('checkpoint')
        download(c['checkpoint']['cap'],checkpoint,size=c['checkpoint']['bytes'],sha256=c['checkpoint']['sha256'],
            absolute_deadline=started+min(240,SCOPE['worker_seconds_max']*.5),cap_validator=validate_flux_checkpoint)
        stage('dependencies')
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        code=child_run(['/isaac-sim/python.sh','-m','pip','install','--isolated','--disable-pip-version-check',
            '--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes',
            '--find-links',str(payload/'deps/wheels'),'--target',str(payload/'dependency-site'),'-r',str(payload/'deps/overlay.lock')],
            out/'install.log',env,started+min(300,SCOPE['worker_seconds_max']*.6))
        if code:raise RuntimeError('Dependency install failed')
        rows=isolate(payload/'formal_data',work/'isolated-clips')
        if [r['skill'] for r in rows]!=SCOPE['skills'] or len(rows)!=SCOPE['max_clips']:
            raise ValueError('Isolated cases exceed approved scope')
        sys.path.insert(0,str(payload))
        from student_adapter.worker_budget import isolated_child_env
        paths=[str(root),str(payload),str(payload/'dependency-site'),str(bundle/'overlay'),str(bundle/'runtime/php'),
            str(bundle/'runtime/holosoma/src/holosoma'),str(bundle),str(payload/'native-source')]
        for i,row in enumerate(rows):
            # Each distinct skill is attempted once; a teardown crash is recorded,
            # never converted to success and never triggers repetition of that skill.
            caseout=out/row['skill'];caseout.mkdir()
            req={k:c[k] for k in ['scope','task_id','_wb_agent','source_plan','source_completed_updates']}
            req.update(output=str(caseout),bundle=str(bundle),data_root=str(payload/'formal_data'),
                teachers=[str(payload/'teachers'/f'{j}.pt') for j in range(5)],checkpoint=str(checkpoint),
                source_checkpoint_sha256=c['checkpoint']['sha256'],obstacle_case=row)
            reqpath=work/(row['skill']+'-request.json');reqpath.write_text(json.dumps(req))
            planpath=work/'source-plan.json'
            if not planpath.exists():planpath.write_text(json.dumps(c['source_plan']))
            case_limit=min(started+SCOPE['worker_seconds_max']-45,time.monotonic()+225)
            child_env=isolated_child_env(os.environ,pythonpath=os.pathsep.join(paths),output=caseout,
                asset_root=bundle/'asset',train_deadline=case_limit-35,task_id=c['task_id'])
            child_env.update(HUD04_VIDEO_REQUEST=str(reqpath),HUD04_FORMAL_PLAN=str(planpath),HUD04_VIDEO_DEADLINE=str(case_limit-35))
            for key in ['ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT']:
                if key in os.environ:child_env[key]=os.environ[key]
            stage(row['skill'])
            code=child_run(['/isaac-sim/python.sh',str(root/'obstacle_video_child.py')],caseout/'native.log',child_env,case_limit,cwd=payload)
            record={'skill':row['skill'],'native_returncode':code,'video_evidence_present':(caseout/'video-evidence.json').is_file()}
            state['cases'].append(record)
            print(json.dumps({'event':'obstacle_case_completed',**record}),flush=True)
        state['worker_passed']=len(state['cases'])==SCOPE['max_clips'] and all(r['native_returncode']==0 and r['video_evidence_present'] for r in state['cases'])
        stage('complete' if state['worker_passed'] else 'completed_with_native_failures')
    except Exception as e:
        state['error_type']=type(e).__name__
        print(json.dumps({'event':'obstacle_video_failure','stage':state['stage'],'error_type':type(e).__name__}),flush=True)
    state['elapsed_seconds']=time.monotonic()-started
    (out/'bootstrap-result.json').write_text(json.dumps(state,indent=2))
    with deadline(started+SCOPE['worker_seconds_max']-20):
        files=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(out.rglob('*')) if f.is_file()]
        if sum(f['bytes'] for f in files)>450*1024**2:raise ValueError('Result size bound')
        (out/'result-manifest.json').write_text(json.dumps({'scope':SCOPE,'files':files},indent=2))
        archive=work/'video-result.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(out))
    receipt=transfer(c['output_cap'],archive,method='PUT',size=archive.stat().st_size,sha256=digest(archive),absolute_deadline=started+SCOPE['worker_seconds_max']-10)
    print(json.dumps({'event':'obstacle_video_result_uploaded',**receipt,'worker_passed':state['worker_passed']}),flush=True)
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print(json.dumps({'event':'obstacle_video_bootstrap_failed','error_type':type(e).__name__}),flush=True);sys.exit(1)
