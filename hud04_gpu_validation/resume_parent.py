"""One approved 3000-env continuation. No account credentials in worker."""
import json,time,zipfile
from pathlib import Path
from scoped_transport import digest,transfer
from resume_contract import SOURCE_SHA,SOURCE_BYTES
import worker_utils

def validate_result(out,rc,task):
    out=Path(out)
    r=json.loads((out/'distillation-result.json').read_text())
    v=json.loads((out/'resume-verification.json').read_text())
    if v.get('source_sha256')!=SOURCE_SHA or v.get('next_iteration')!=5 or not v.get('model_optimizer_scalars_readiness_equal'):
        raise ValueError('Fresh process restore not verified')
    if not 1<=r.get('completed_updates',0)<=1000 or r.get('last_completed_iteration')!=4+r['completed_updates']:
        raise ValueError('Resumed update bounds/counters mismatch')
    if r.get('algorithm')!='dagger_ppo' or not all(r.get('modules_changed',{}).get(k) is True for k in ('student','critic','depth_backbone')):
        raise ValueError('Missing resumed learning')
    if not all(r.get(k) is True for k in ('teachers_frozen','native_reload_verified','fresh_process_resume_verified')):
        raise ValueError('Checkpoint acceptance incomplete')
    if len(r.get('valid_labels_by_skill',[]))!=3 or not all(x>0 for x in r['valid_labels_by_skill']):raise ValueError('Routes incomplete')
    p=out/'student_dagger_ppo_validation.pt'
    if p.is_symlink() or not p.is_file() or digest(p)!=r['checkpoint_sha256']:raise ValueError('Checkpoint digest mismatch')
    life=json.loads((out/'lifecycle.json').read_text())
    prov=json.loads((out/'provenance.json').read_text())
    if prov.get('task_id')!=task or prov.get('num_envs')!=3000 or life.get('task_id')!=task:raise ValueError('Provenance mismatch')
    if rc!=0 or not all(life.get(k) is True for k in ('train_returned','close_returned','child_returned','fast_shutdown_disabled_verified')):
        raise ValueError('Native lifecycle failed; checkpoint remains diagnostic')
    return dict(completed_updates=r['completed_updates'],fresh_process_resume_verified=True,checkpoint_sha256=r['checkpoint_sha256'])

def pack(work,state):
    (work/'bootstrap-status.json').write_text(json.dumps(state,indent=2))
    out=work/'native-output';final=out/'student_dagger_ppo_validation.pt'
    checkpoints=sorted(out.rglob('model_*.pt'),key=lambda p:p.stat().st_mtime)
    keep={final} if final.is_file() else set(checkpoints[-1:])
    rows=[];total=0
    candidates=[work/'bootstrap-status.json',work/'install.log',work/'gpu-preflight.log',work/'git-input-verification.json']+list(out.rglob('*'))
    for p in sorted(set(candidates)):
        if p.is_symlink() or not p.is_file():continue
        if p.suffix=='.pt' and p not in keep:continue
        if p.suffix not in ('.pt','.json','.jsonl','.log','.yaml','.yml') and 'tfevents' not in p.name:continue
        total+=p.stat().st_size
        if total>160*1024**2:raise ValueError('Result size bound exceeded')
        rows.append(dict(path=p.relative_to(work).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    manifest=dict(task_id=state['task_id'],source_commit=state['bootstrap_commit'],files=rows,
        num_envs=3000,max_updates=1000,training_seconds_max=6600,worker_seconds_max=7200,
        resume_source_sha256=SOURCE_SHA,automatic_retries=0)
    archive=work/'flux-result.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('result-manifest.json',json.dumps(manifest,indent=2))
        for r in rows:z.write(work/r['path'],r['path'])
    return archive

def main():
    source=Path(__file__).with_name('three_teacher_parent.py').read_text()
    changes={
      "'output-cap','thread-id'":"'output-cap','resume-cap','relay-cap','thread-id'",
      "caps={'output':json.loads(base64.urlsafe_b64decode(a.output_cap))}":"caps={'output':json.loads(base64.urlsafe_b64decode(a.output_cap)),'resume':json.loads(base64.urlsafe_b64decode(a.resume_cap)),'relay':json.loads(base64.urlsafe_b64decode(a.relay_cap))}",
      "time.time()+1800":"time.time()+300",
      "target=unused":"target=unused"
    }
    changes.pop('target=unused')
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('Parent binding anchor changed')
        source=source.replace(old,new,1)
    for old,new in [('num_envs=6','num_envs=3000'),('max_updates=5','max_updates=1000'),('training_seconds_max=600','training_seconds_max=6600'),('worker_seconds_max=1800','worker_seconds_max=7200')]:source=source.replace(old,new)
    changes={
      "state['stage']='git_input_reassembly'":"resume=work/'resume.pt'\n        transfer(caps['resume'],resume,method='GET',size=47769831,sha256='"+SOURCE_SHA+"',absolute_deadline=start+300)\n        state['stage']='git_input_reassembly'",
      'min(start+1530,time.monotonic()+540)':'min(start+6900,time.monotonic()+6600)',
      "root/'three_teacher_child.py'":"root/'resume_child.py'",
      "'--train-deadline',str(train_deadline)]":"'--train-deadline',str(train_deadline),'--resume-checkpoint',str(resume)]",
      "rc=stage(cmd,out/'native.log',env,min(start+1650,train_deadline+60))":"rc=stage(cmd,out/'native.log',env,min(start+7050,train_deadline+60),stop_file=work/'stop-request',soft=train_deadline)",
      'absolute_deadline=start+1790':'absolute_deadline=start+7190',
      "threading.Thread(target=watchdog,args=(start,),daemon=True).start()":"threading.Thread(target=watchdog,args=(start+5400,),daemon=True).start()\n    from capability_relay import poll\n    threading.Thread(target=poll,args=(caps,start+7200),daemon=True).start()"
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('Parent execution anchor changed')
        source=source.replace(old,new,1)
    namespace={'__file__':__file__,'__name__':'resume_bound_parent'}
    exec(compile(source,'audited_parent_with_resume_bounds','exec'),namespace)
    namespace.update(validate_result=validate_result,pack=pack)
    return namespace['main']()

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        worker_utils.emit('bootstrap_failure',error_type=type(e).__name__)
        raise SystemExit(1)