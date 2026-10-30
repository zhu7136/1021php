"""One approved GPU task: five distinct frozen student cases, no training/retry."""
import argparse,json,os,signal,subprocess,sys,threading,time,zipfile
from pathlib import Path
from scoped_transport import digest,deadline,transfer,validate_capability
from range_input import download
from checkpoint_source import validate_flux_checkpoint
from local_trial_inputs import materialize
from bounded_training_trial import unpack
from inline_execution import decode_inline
from five_skill_inputs import SCOPE,check_config,isolate,case_deadline

def emit(event,**values):print(json.dumps({'event':event,**values},allow_nan=False),flush=True)

def main():
    started=time.monotonic();active=[None];lock=threading.Lock()
    def guard():
        time.sleep(max(0,started+900-time.monotonic()))
        with lock:
            child=active[0]
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        os._exit(124)
    threading.Thread(target=guard,daemon=True).start()
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--task-id',required=True)
    args=parser.parse_args();_,config=decode_inline(args.config,args.sha256)
    check_config(config,args.task_id)
    root=Path(__file__).resolve().parent
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()!=config['code_commit']:
        raise ValueError('Commit pin')
    validate_capability(config['output_cap'],'PUT');validate_flux_checkpoint(config['checkpoint']['cap'],'GET')
    work=root/'private-five-skill-screen';work.mkdir(exist_ok=False)
    out=work/'output';out.mkdir()
    state={'schema':'hud04-five-skill-screen-result-v1','task_id':args.task_id,'scope':SCOPE,
        '_wb_agent':config['_wb_agent'],'code_commit':config['code_commit'],
        'checkpoint_sha256':config['checkpoint']['sha256'],'source_completed_updates':3686,
        'training_updates':0,'automatic_retries':0,'worker_passed':False,
        'five_skill_screen_complete':False,'task_success':None,'skills_evaluated':False,
        'stage':'inputs','cases':[]}
    def stage(value):
        state['stage']=value;emit('five_skill_stage',stage=value)
    def child_run(command,path,env,limit,cwd=None):
        if time.monotonic()>=limit:raise TimeoutError('No remaining child budget')
        with path.open('xb') as log:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[0]=child
            timed_out=False
            try:
                try:code=child.wait(timeout=max(.01,limit-time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out=True
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    code=child.wait(timeout=5)
            finally:
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait(timeout=5)
                with lock:active[0]=None
            return code,timed_out
    try:
        materialize(root/'inputs',work,manifest_sha256=config['shard_manifest_sha256'],
                    absolute_deadline=started+180,unpack_archive=unpack)
        payload=work/'payload';bundle=payload/'bundle';checkpoint=work/'source-final.pt'
        stage('checkpoint')
        download(config['checkpoint']['cap'],checkpoint,size=config['checkpoint']['bytes'],
                 sha256=config['checkpoint']['sha256'],absolute_deadline=started+240,
                 cap_validator=validate_flux_checkpoint)
        stage('dependencies')
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        code,timed_out=child_run(['/isaac-sim/python.sh','-m','pip','install','--isolated',
            '--disable-pip-version-check','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:',
            '--require-hashes','--find-links',str(payload/'deps/wheels'),'--target',str(payload/'dependency-site'),
            '-r',str(payload/'deps/overlay.lock')],out/'install.log',env,started+300)
        if code or timed_out:raise RuntimeError('Dependency installation failed')
        stage('isolate_five_skills')
        with deadline(started+330):rows=isolate(payload/'formal_data',work/'isolated-clips')
        if [r['skill'] for r in rows]!=SCOPE['skills']:raise ValueError('Skill identity/order')
        sys.path.insert(0,str(payload))
        from student_adapter.worker_budget import isolated_child_env
        paths=[str(root),str(payload),str(payload/'dependency-site'),str(bundle/'overlay'),
            str(bundle/'runtime/php'),str(bundle/'runtime/holosoma/src/holosoma'),str(bundle),str(payload/'native-source')]
        planpath=work/'source-plan.json';planpath.write_text(json.dumps(config['source_plan']))
        for index,row in enumerate(rows):
            limit=case_deadline(started,time.monotonic(),len(rows)-index)
            if limit is None:
                for remaining in rows[index:]:
                    record={'skill':remaining['skill'],'attempted':False,'reason':'worker_budget_exhausted','task_success':None}
                    state['cases'].append(record);emit('five_skill_case_completed',**record)
                break
            caseout=out/row['skill'];caseout.mkdir()
            req={k:config[k] for k in ['scope','task_id','_wb_agent','source_plan','source_completed_updates',
                 'evaluation_role','evaluation_seed','same_state_action_probe','depth_reset_diagnostics']}
            req.update(output=str(caseout),bundle=str(bundle),data_root=str(payload/'formal_data'),
                teachers=[str(payload/'teachers'/f'{j}.pt') for j in range(5)],checkpoint=str(checkpoint),
                source_checkpoint_sha256=config['checkpoint']['sha256'],obstacle_case=row)
            reqpath=work/(row['skill']+'-request.json');reqpath.write_text(json.dumps(req))
            childenv=isolated_child_env(os.environ,pythonpath=os.pathsep.join(paths),output=caseout,
                asset_root=bundle/'asset',train_deadline=limit-25,task_id=args.task_id)
            childenv.update(HUD04_VIDEO_REQUEST=str(reqpath),HUD04_FORMAL_PLAN=str(planpath),HUD04_VIDEO_DEADLINE=str(limit-25))
            for key in ['ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH','CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT']:
                if key in os.environ:childenv[key]=os.environ[key]
            stage(row['skill'])
            code,timed_out=child_run(['/isaac-sim/python.sh',str(root/'hud04_eval_child.py')],
                                     caseout/'native.log',childenv,limit,cwd=payload)
            record={'skill':row['skill'],'attempted':True,'native_returncode':code,
                    'child_timed_out':timed_out,'task_success':None,'diagnostic_valid':False}
            result=caseout/'single-episode-result.json'
            if result.is_file():
                diagnostic=json.loads(result.read_text())
                # The result must refer to this exact child, source and role.
                if (diagnostic.get('source_checkpoint_sha256')!=config['checkpoint']['sha256']
                    or diagnostic.get('source_completed_updates')!=3686 or diagnostic.get('role')!='student'
                    or diagnostic.get('case',{}).get('skill')!=row['skill']
                    or diagnostic.get('training_updates')!=0):raise ValueError('Child diagnostic identity')
                record.update(diagnostic_outcome=diagnostic['outcome'],policy_steps=diagnostic['policy_steps'],
                    simulated_policy_seconds=diagnostic['simulated_policy_seconds'],
                    model_parameters_unchanged=diagnostic['model_parameters_unchanged'],
                    terminal_observation=diagnostic.get('terminal_observation'),
                    diagnostic_valid=not diagnostic['outcome'].startswith('engineering_invalid')
                                     and diagnostic['model_parameters_unchanged'])
            state['cases'].append(record)
            (out/'screen-progress.json').write_text(json.dumps(state,indent=2,allow_nan=False))
            emit('five_skill_case_completed',**record)
        state['five_skill_screen_complete']=len(state['cases'])==5 and all(r.get('diagnostic_valid',False) for r in state['cases'])
        state['worker_passed']=state['five_skill_screen_complete'] and all(
            r.get('native_returncode')==0 and not r.get('child_timed_out') for r in state['cases'])
        stage('screen_complete' if state['five_skill_screen_complete'] else 'screen_incomplete')
    except Exception as error:
        # Never emit arbitrary exception arguments containing private URLs.
        state['error_type']=type(error).__name__
        emit('five_skill_failure',stage=state['stage'],error_type=type(error).__name__)
    state['elapsed_seconds']=time.monotonic()-started
    (out/'five-skill-result.json').write_text(json.dumps(state,indent=2,allow_nan=False))
    emit('five_skill_summary',cases_reported=len(state['cases']),five_skill_screen_complete=state['five_skill_screen_complete'],
         worker_passed=state['worker_passed'],training_updates=0,elapsed_seconds=state['elapsed_seconds'])
    with deadline(started+870):
        files=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(out.rglob('*')) if p.is_file()]
        if sum(p['bytes'] for p in files)>450*1024**2:raise ValueError('Result size bound')
        (out/'result-manifest.json').write_text(json.dumps({'scope':SCOPE,'files':files},indent=2))
        archive=work/'five-skill-result.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as output:
            for file in sorted(out.rglob('*')):
                if file.is_file():output.write(file,file.relative_to(out))
    receipt=transfer(config['output_cap'],archive,method='PUT',size=archive.stat().st_size,
        sha256=digest(archive),absolute_deadline=started+890)
    emit('five_skill_result_uploaded',**receipt,worker_passed=state['worker_passed'])
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        emit('five_skill_worker_failed',error_type=type(error).__name__);sys.exit(1)