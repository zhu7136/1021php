"""Approved one-shot Hu-D04 DAgger+PPO validation. No account credentials."""
import argparse,base64,json,os,re,stat,subprocess,sys,threading,time,zipfile
from pathlib import Path,PurePosixPath
from bootstrap_base import safe_extract,clean_env
from scoped_transport import digest,transfer,validate_capability
from range_input import download
from worker_utils import stage,watchdog,emit

INPUTS={
 'base':(123410856,'490d7d1b8b927f42303816c1a40f440c05ef5393aec950724ab897fad45c643f'),
 'data':(470500407,'f2e3c5de404b68b9ac1e6849e9d4f02aa5fcf88bc378cc45dc2b8865ac3aea79')}

def extract_data(archive,destination):
    destination=Path(destination);destination.mkdir(exist_ok=False);seen=set()
    with zipfile.ZipFile(archive) as z:
        if len(z.infolist())>20 or sum(i.file_size for i in z.infolist())>800*1024**2:
            raise ValueError('Dataset expansion bounds')
        for i in z.infolist():
            p=PurePosixPath(i.filename);mode=i.external_attr>>16
            if (p.is_absolute() or '..' in p.parts or '\\' in i.filename or i.filename in seen
                or len(p.parts)<2 or p.parts[0]!='data' or i.is_dir() or stat.S_ISLNK(mode)
                or stat.S_IFMT(mode) not in (0,stat.S_IFREG)):
                raise ValueError('Unsafe dataset archive member')
            seen.add(i.filename);target=destination.joinpath(*p.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as src,target.open('xb') as out:
                while True:
                    chunk=src.read(1024**2)
                    if not chunk:break
                    out.write(chunk)
    return destination/'data'

def validate_result(out,rc,task):
    out=Path(out)
    if rc!=0:raise ValueError('Native process did not exit cleanly')
    r=json.loads((out/'distillation-result.json').read_text());life=json.loads((out/'lifecycle.json').read_text())
    prov=json.loads((out/'provenance.json').read_text())
    if r.get('algorithm')!='dagger_ppo' or r.get('completed_updates')!=5:
        raise ValueError('Native five-update PPO acceptance missing')
    if not all(r.get('modules_changed',{}).get(k) is True for k in ('student','critic','depth_backbone')) or not all(r.get(k) is True for k in ('teachers_frozen','native_reload_verified')):
        raise ValueError('Model/optimizer verification incomplete')
    if len(r.get('valid_labels_by_skill',[]))!=3 or not all(n>0 for n in r['valid_labels_by_skill']):
        raise ValueError('All three teacher routes must be observed')
    if life.get('task_id')!=task or not all(life.get(k) is True for k in ('train_returned','close_returned','child_returned','fast_shutdown_disabled_verified')):
        raise ValueError('Real simulator lifecycle incomplete')
    if prov.get('task_id')!=task or prov.get('algorithm')!='dagger_ppo' or prov.get('num_envs')!=6:
        raise ValueError('Result provenance mismatch')
    final=out/'student_dagger_ppo_validation.pt'
    if final.is_symlink() or not final.is_file() or digest(final)!=r.get('checkpoint_sha256'):
        raise ValueError('Final checkpoint binding mismatch')
    return dict(completed_updates=5,checkpoint_sha256=digest(final),native_full_state_reload_verified=True)

def pack(work,state):
    (work/'bootstrap-status.json').write_text(json.dumps(state,indent=2));rows=[];total=0
    candidates=[work/'bootstrap-status.json']+list((work/'native-output').rglob('*'))+[work/'install.log',work/'gpu-preflight.log']
    for p in sorted(set(candidates)):
        if not p.is_file() or p.is_symlink():continue
        if p.suffix=='.pt' and p.name!='student_dagger_ppo_validation.pt':continue
        if p.suffix not in ('.pt','.json','.jsonl','.log','.yaml','.yml') and 'tfevents' not in p.name:continue
        total+=p.stat().st_size
        if total>160*1024**2:raise ValueError('Bounded result archive exceeded')
        rows.append(dict(path=p.relative_to(work).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    manifest=dict(task_id=state['task_id'],source_commit=state['bootstrap_commit'],files=rows,
        num_envs=6,max_updates=5,training_seconds_max=600,worker_seconds_max=1800,automatic_retries=0)
    result=work/'flux-result.zip'
    with zipfile.ZipFile(result,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('result-manifest.json',json.dumps(manifest,indent=2))
        for r in rows:z.write(work/r['path'],r['path'])
    return result

def main(argv=None):
    start=time.monotonic();p=argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true')
    for n in ('task-id','commit','base-cap','data-cap','output-cap','thread-id','turn-id'):p.add_argument('--'+n)
    a=p.parse_args(argv)
    if not a.execute:emit('execution_not_requested',gpu_started=False,num_envs=6,max_updates=5,training_seconds_max=600,worker_seconds_max=1800,automatic_retries=0);return 0
    if not a.task_id or not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id) or not re.fullmatch('[0-9a-f]{40}',a.commit or '') or not all(getattr(a,n) for n in ('base_cap','data_cap','output_cap','thread_id','turn_id')):
        raise ValueError('Task/commit/provenance binding required')
    caps={n:json.loads(base64.urlsafe_b64decode(getattr(a,n+'_cap'))) for n in ('base','data','output')}
    for n,c in caps.items():
        _,expiry=validate_capability(c,'PUT' if n=='output' else 'GET')
        if expiry<time.time()+1800:raise ValueError('Capability must cover bounded worker lifetime')
    root=Path(__file__).resolve().parent;env=clean_env()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=env,text=True,timeout=5).strip()!=a.commit:
        raise ValueError('Pinned checkout mismatch')
    threading.Thread(target=watchdog,args=(start,),daemon=True).start()
    work=root/('private-validation-'+a.task_id);work.mkdir(exist_ok=False)
    state=dict(task_id=a.task_id,bootstrap_commit=a.commit,worker_seconds_max=1800,training_seconds_max=600,
        requested_num_envs=6,max_updates=5,automatic_retries=0,worker_passed=False,native_training_started=False,
        algorithm='dagger_ppo',skill_success_verified=False,account_credentials_forwarded=False,stage='inputs')
    try:
        paths={}
        for n,(size,sha) in INPUTS.items():
            emit('hud04_stage',stage='download_'+n,task_id=a.task_id)
            paths[n]=work/(n+'.zip');progress={}
            download(caps[n],paths[n],size=size,sha256=sha,absolute_deadline=start+600,progress=progress)
        payload=work/'payload';safe_extract(paths['base'],payload)
        data=extract_data(paths['data'],work/'dataset');state['inputs_hash_verified']=True
        vendor='/isaac-sim/python.sh';site=work/'dependency-site';state['stage']='install'
        emit('hud04_stage',stage=state['stage'],task_id=a.task_id)
        if not Path(vendor).is_file():raise ValueError('Pinned simulator interpreter unavailable')
        if stage([vendor,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')],work/'install.log',env,start+720):
            raise ValueError('Offline installation failed')
        env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','');state['stage']='gpu_preflight'
        probe="import torch,numpy;p=torch.cuda.get_device_properties(0);print(p.name,p.total_memory,flush=True);assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4' and torch.cuda.device_count()==1 and '4090' in p.name and [p.major,p.minor]==[8,9] and 23<=p.total_memory/2**30<=25"
        if stage([vendor,'-c',probe],work/'gpu-preflight.log',env,start+750):raise ValueError('GPU contract mismatch')
        if time.monotonic()>start+750:raise TimeoutError('Late training start rejected')
        # Reserve 60s for real native cleanup inside the approved 600s phase.
        # Parent forcibly ends the entire native child (including startup and
        # cleanup) after 600s; the learner receives at most 540s.
        train_deadline=min(start+1530,time.monotonic()+540)
        out=work/'native-output';out.mkdir();env['HUD04_FLUX_TASK_ID']=a.task_id
        state.update(stage='native_training',native_training_started=True,training_deadline_monotonic=train_deadline)
        emit('hud04_stage',stage=state['stage'],task_id=a.task_id)
        cmd=[vendor,str(root/'three_teacher_child.py'),'--payload',str(payload),'--data-root',str(data),
            '--output',str(out),'--site',str(site),'--thread-id',a.thread_id,'--turn-id',a.turn_id,'--train-deadline',str(train_deadline)]
        rc=stage(cmd,out/'native.log',env,min(start+1650,train_deadline+60))
        state.update(native_exit_code=rc,stage='result_validation');state.update(validate_result(out,rc,a.task_id))
        state.update(worker_passed=True,stage='complete')
    except Exception as e:
        state['error_type']=type(e).__name__;emit('stage_failure',stage=state['stage'],error_type=type(e).__name__)
    state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3)
    archive=pack(work,state)
    receipt=transfer(caps['output'],archive,method='PUT',size=archive.stat().st_size,sha256=digest(archive),absolute_deadline=start+1790)
    emit('result_uploaded',task_id=a.task_id,bytes=receipt['bytes'],sha256=receipt['sha256'],worker_passed=state['worker_passed'])
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:emit('bootstrap_failure',error_type=type(e).__name__);sys.exit(1)