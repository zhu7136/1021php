"""Single original-task continuation. Needs pinned code + fresh scoped GCS tickets.

No account credentials go to the simulator. No task create/edit/run calls here.
"""
import argparse,base64,hashlib,json,math,os,re,shutil,subprocess,sys,threading,time
from pathlib import Path
from bootstrap_base import safe_extract,clean_env
from scoped_transport import transfer,digest,validate_capability
from range_input import download
from worker_utils import stage,pack,watchdog,emit
from budget_policy import deadlines,require_training_window
TASK='TASK_20261029_094'
INPUTS={'base':(113889241,'6ed95a54d8846611b8e2519dcc014a9aec9c5dd885914d15137cc18e8f1473e0'),'input':(91493471,'700461597f20cf2adad807a92f7996aed9eebba4a853e8444e5eb3d92b6bbb3e'),'checkpoint':(10344315,'7fc623a03fabd6920c122d844ceaa91e1a5fab29a263e4bb9312d5851e623164')}

def validate_result(out,rc,task=TASK):
    out=Path(out).resolve()
    if rc!=0 or task!=TASK:raise ValueError('Native process/task mismatch')
    def read(name):return json.loads((out/name).read_text())
    source=read('source_resume_verified.json');continuation=read('continuation_state.json');learning=read('learning_state.json');reload=read('native_reload.json');life=read('lifecycle.json');progress=read('progress.json');provenance=read('provenance.json')
    if source.get('source_sha256')!=INPUTS['checkpoint'][1] or source.get('next_iteration')!=660 or source.get('source_iteration')!=659 or not all(source.get(k) is True for k in ('model_and_optimizers_exact','learning_rates_exact')):raise ValueError('Source restore evidence missing')
    count=continuation.get('new_completed_updates')
    if type(count) is not int or not 1<=count<=100000:raise ValueError('No bounded continuation updates')
    if continuation.get('fresh_training') is not False or continuation.get('source_iteration')!=659 or continuation.get('first_new_iteration')!=660 or continuation.get('last_iteration')!=659+count or continuation.get('model_optimizer_resume') is not True:raise ValueError('Continuation iteration mismatch')
    if any(d.get('completed_updates')!=count for d in (learning,reload,progress)):raise ValueError('Update counts differ')
    if progress.get('last_iteration_index')!=659+count or reload.get('iteration')!=659+count:raise ValueError('Final iteration evidence mismatch')
    if learning.get('fresh_training') is not False or learning.get('new_process_resume_verified') is not True or not all(learning.get(k) is True for k in ('actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite')):raise ValueError('Native learning evidence incomplete')
    if reload.get('native_load_passed') is not True or not {'actor_optimizer','critic_optimizer'}<=set(reload.get('checked_optimizers',[])):raise ValueError('Final native optimizer reload missing')
    if life.get('task_id')!=task or not all(life.get(k) is True for k in ('close_returned','train_returned','child_returned','fast_shutdown_disabled_verified')):raise ValueError('Normal close evidence missing')
    if provenance.get('task_id')!=task or provenance.get('fresh_training') is not False or provenance.get('source_checkpoint_sha256')!=INPUTS['checkpoint'][1]:raise ValueError('Provenance mismatch')
    losses=progress.get('losses',{})
    if not losses or not all(math.isfinite(float(v)) for v in losses.values()):raise ValueError('Nonfinite or missing losses')
    rel=Path(reload['path'])
    if rel.is_absolute() or '..' in rel.parts or rel.name!='model_smoke_final.pt' or any(p.is_symlink() for p in [out/rel,*(out/rel).parents]):raise ValueError('Unsafe final checkpoint path')
    final=out/rel
    if not final.resolve().is_relative_to(out) or not final.is_file() or final.stat().st_size==0 or learning.get('checkpoint_relative_path')!=rel.as_posix() or digest(final)!=learning.get('checkpoint_sha256'):raise ValueError('Final checkpoint binding mismatch')
    return {'completed_updates':count,'last_iteration':659+count,'checkpoint_sha256':digest(final),'close_returned':True,'new_process_resume_verified':True}

def prepare_inputs(base_zip,overlay_zip,checkpoint,payload,overlay):
    for name,path in [('base',base_zip),('input',overlay_zip),('checkpoint',checkpoint)]:
        if Path(path).stat().st_size!=INPUTS[name][0] or digest(path)!=INPUTS[name][1]:raise ValueError('Exact input identity mismatch')
    base=safe_extract(base_zip,payload);motion=safe_extract(overlay_zip,overlay)
    if base['scope'].get('runtime_only') is not True or base['scope'].get('checkpoint_files')!=0:raise ValueError('Runtime-only base required')
    scope=motion['scope'];cfg=json.loads((overlay/'teacher-code/config.json').read_text());cases=json.loads((overlay/'teacher-code/cases.json').read_text())
    if scope['skill']!='low-step' or len(cases)!=56 or sum(c['frames'] for c in cases)!=25676 or cfg['training']['num_envs']!=4096 or any(cfg['training'][k] is not None for k in ('checkpoint','teacher_checkpoint','registry_name','preprocess_hook')):raise ValueError('Original full-skill input mismatch')
    for row in motion['files']:
        target=payload/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Refuse inherited input overwrite')
        shutil.copyfile(overlay/row['path'],target)
    dest=payload/'continuation-source';dest.mkdir();shutil.copyfile(checkpoint,dest/'source-checkpoint.pt')
    return dest/'source-checkpoint.pt'

def main(argv=None):
    start=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    for name in ('task-id','commit','base-cap','input-cap','checkpoint-cap','output-cap','thread-id','turn-id'):p.add_argument('--'+name)
    a=p.parse_args(argv)
    if not a.execute:emit('execution_not_requested',gpu_started=False,worker_seconds_max=1800,automatic_retries=0);return 0
    if a.task_id!=TASK or not a.commit or not re.fullmatch('[0-9a-f]{40}',a.commit) or not all(getattr(a,n) for n in ('base_cap','input_cap','checkpoint_cap','output_cap','thread_id','turn_id')):raise ValueError('Missing execution binding')
    ends=deadlines(start,time.monotonic());caps={n:json.loads(base64.urlsafe_b64decode(getattr(a,n+'_cap'))) for n in ('base','input','checkpoint','output')}
    for n,c in caps.items():
        _,expiry=validate_capability(c,'PUT' if n=='output' else 'GET')
        if expiry<time.time()+1800:raise ValueError('Ticket must cover whole worker budget')
    root=Path(__file__).resolve().parent;env=clean_env()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=env,text=True,timeout=5).strip()!=a.commit:raise ValueError('Pinned code mismatch')
    threading.Thread(target=watchdog,args=(start,),daemon=True).start()
    work=root/('private-resume-'+a.task_id);work.mkdir(exist_ok=False)
    state={'task_id':a.task_id,'bootstrap_commit':a.commit,'worker_seconds_max':1800,'automatic_retries':0,'requested_num_envs':4096,'fresh_training':False,'teacher_converged':False,'worker_passed':False,'native_training_started':False,'stage':'inputs'}
    try:
        paths={}
        for n,(size,sha) in INPUTS.items():
            paths[n]=work/({'base':'input.zip','input':'full-input.zip','checkpoint':'source-checkpoint.pt'}[n])
            download(caps[n],paths[n],size=size,sha256=sha,absolute_deadline=ends['input'])
        payload=work/'payload';source=prepare_inputs(paths['base'],paths['input'],paths['checkpoint'],payload,work/'overlay')
        state['input_verified']=True;state['stage']='install'
        vendor='/isaac-sim/python.sh';site=work/'dependency-site'
        if not Path(vendor).is_file():raise ValueError('Vendor interpreter unavailable')
        if stage([vendor,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')],work/'install.log',env,ends['install']):raise ValueError('Offline install failed')
        env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','');state['stage']='gpu_preflight'
        probe="import torch,numpy;p=torch.cuda.get_device_properties(0);print(p.name,p.total_memory,flush=True);assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4' and torch.cuda.device_count()==1 and '4090' in p.name and [p.major,p.minor]==[8,9] and 23<=p.total_memory/2**30<=25"
        if stage([vendor,'-c',probe],work/'gpu-preflight.log',env,ends['gpu']):raise ValueError('GPU contract failed')
        require_training_window(start,time.monotonic());out=work/'native-output';out.mkdir();env.update(PHP_FLUX_TASK_ID=a.task_id,PHP_FLUX_STOP_FILE=str(out/'STOP_AFTER_UPDATE'))
        state.update(stage='native_training',native_training_started=True)
        cmd=[vendor,str(root/'low_step_resume_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--thread-id',a.thread_id,'--turn-id',a.turn_id,'--source-checkpoint',str(source)]
        rc=stage(cmd,out/'native.log',env,ends['train_hard'],stop_file=out/'STOP_AFTER_UPDATE',soft=ends['train_soft'])
        state.update(native_exit_code=rc,stage='result_validation');state.update(validate_result(out,rc,a.task_id));state.update(worker_passed=True,stage='complete')
    except Exception as e:
        state['error_type']=type(e).__name__;emit('stage_failure',stage=state['stage'],error_type=type(e).__name__)
    state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3);result=pack(work,state)
    receipt=transfer(caps['output'],result,method='PUT',size=result.stat().st_size,sha256=digest(result),absolute_deadline=ends['result'])
    emit('result_uploaded',task_id=a.task_id,bytes=receipt['bytes'],sha256=receipt['sha256'],worker_passed=state['worker_passed'])
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:emit('bootstrap_failure',error_type=type(e).__name__);sys.exit(1)