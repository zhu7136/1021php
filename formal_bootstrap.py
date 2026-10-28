"""One authorized Hu-D04 4096env continuation, <=1800s worker, no retries."""
import argparse,base64,hashlib,json,os,signal,subprocess,sys,threading,time,zipfile
from pathlib import Path
from bootstrap_base import safe_extract,clean_env
from scoped_transport import transfer,digest
from qq_input import validate_capability
import transfer_diagnostics
transfer_diagnostics.validate_capability=validate_capability
from transfer_diagnostics import download
ACTIVE=set()
SOURCE_RESULT_SHA='02368327e9ddc4aae47862fbeeebec3a9fbc80e4f19cd320752c3f4d02e7449b'
CHECKPOINT_SHA='aa543b193b8502142d03d4cbdaba0cb10a0909f2140c8c6a9e8d58b682e7cf1a'

def emit(event,**fields):print(json.dumps({'event':event,**fields}),flush=True)

def stage(command,log,env,hard,stop_file=None,soft=None):
    if time.monotonic()>=hard:raise TimeoutError('No stage budget')
    pos=0;last=0
    with Path(log).open('xb') as f:
        child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);ACTIVE.add(child.pid)
        try:
            while child.poll() is None:
                now=time.monotonic()
                if stop_file and now>=soft:Path(stop_file).touch(exist_ok=True)
                if now>=hard:raise TimeoutError('Stage deadline')
                if stop_file and now-last>=5:
                    with Path(log).open('r',errors='replace') as read:
                        read.seek(pos);lines=read.readlines(262144);pos=read.tell()
                    selected={}
                    for line in lines:
                        if not line.startswith('{'):continue
                        try:obj=json.loads(line)
                        except ValueError:continue
                        if obj.get('event') in ['formal_update','capacity_initialized']:selected[obj['event']]=obj
                    for obj in selected.values():
                        safe={k:v for k,v in obj.items() if isinstance(v,(int,float,bool)) or k=='event'}
                        emit('native_progress',**{'native_event':obj['event'],**{k:v for k,v in safe.items() if k!='event'}})
                    last=now
                time.sleep(.1)
            return child.returncode
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=2);ACTIVE.discard(child.pid)

def watchdog(start):
    time.sleep(max(0,start+1798-time.monotonic()))
    for pid in list(ACTIVE):
        try:os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    os._exit(124)

def pack(work,state):
    (work/'bootstrap-status.json').write_text(json.dumps(state,indent=2))
    checkpoints=sorted((work/'native-output').rglob('model_*.pt'),key=lambda p:p.stat().st_mtime) if (work/'native-output').exists() else []
    keep=set(checkpoints[-3:]);keep.update(p for p in checkpoints if p.name=='model_smoke_final.pt')
    rows=[];total=0
    for p in sorted(work.rglob('*')):
        rel=p.relative_to(work)
        if p.is_symlink() or not p.is_file() or rel.parts[0] in {'payload','dependency-site'}:continue
        if p.name in {'input.zip','source-result.zip','flux-result.zip'}:continue
        if p.suffix=='.pt' and p not in keep:continue
        if p.suffix not in {'.pt','.json','.jsonl','.log','.yaml','.yml'} and 'tfevents' not in p.name:continue
        total+=p.stat().st_size
        if total>160*1024**2:raise ValueError('Result size exceeded')
        rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':digest(p)})
    manifest={'task_id':state['task_id'],'source_commit':state['bootstrap_commit'],'files':rows,'automatic_retries':0,'worker_seconds_max':1800,'requested_num_envs':4096}
    archive=work/'flux-result.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('result-manifest.json',json.dumps(manifest,indent=2))
        for row in rows:z.write(work/row['path'],row['path'])
    return archive

def install_checkpoint(source_result,payload):
    assert digest(source_result)==SOURCE_RESULT_SHA
    with zipfile.ZipFile(source_result) as z:
        m=json.loads(z.read('result-manifest.json'))
        assert m['task_id']=='TASK_20261028_110' and m['source_commit']=='dceade2b9d767ddc30a8f0cf0b301d2f525e009b'
        reload=json.loads(z.read('native-output/native_reload.json'))
        name='native-output/'+reload['path']
        assert name.endswith('/model_smoke_final.pt') and reload['iteration']==128
        row=next(r for r in m['files'] if r['path']==name);b=z.read(name)
        assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']==CHECKPOINT_SHA
        dest=payload/'formal-resume';dest.mkdir()
        with (dest/'source-checkpoint.pt').open('xb') as f:f.write(b)

def main():
    start=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    for name in ['task-id','commit','input-cap','output-cap','checkpoint-cap','thread-id','turn-id']:p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if not a.execute:emit('execution_not_requested');return 0
    threading.Thread(target=watchdog,args=(start,),daemon=True).start()
    caps={name:json.loads(base64.urlsafe_b64decode(getattr(a,name+'_cap'))) for name in ['input','output','checkpoint']}
    for name,method in [('input','GET'),('checkpoint','GET'),('output','PUT')]:validate_capability(caps[name],method)
    root=Path(__file__).resolve().parent;env=clean_env()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=env,text=True,timeout=5).strip()
    assert head==a.commit
    work=root/('private-'+a.task_id);work.mkdir(exist_ok=False)
    state={'task_id':a.task_id,'bootstrap_commit':head,'worker_seconds_max':1800,'automatic_retries':0,
           'requested_num_envs':4096,'worker_passed':False,'teacher_converged':False,'native_training_started':False,'stage':'input'}
    try:
        emit('stage',stage='input')
        state['input_transfer']={}
        download(caps['input'],work/'input.zip',size=123410856,sha256='490d7d1b8b927f42303816c1a40f440c05ef5393aec950724ab897fad45c643f',absolute_deadline=start+120,progress=state['input_transfer'])
        payload=work/'payload';manifest=safe_extract(work/'input.zip',payload)
        assert manifest['scope']['worker_seconds_max']==600 and manifest['scope']['num_envs']==128
        state['stage']='latest_checkpoint';emit('stage',stage='latest_checkpoint')
        state['checkpoint_transfer']={}
        download(caps['checkpoint'],work/'source-result.zip',size=9543100,sha256=SOURCE_RESULT_SHA,
                 absolute_deadline=min(start+420,time.monotonic()+300),progress=state['checkpoint_transfer'])
        install_checkpoint(work/'source-result.zip',payload)
        # Only credential-free code from the pinned repository is overlaid; base config/assets remain unchanged.
        for name in ['formal_child.py','formal_algo.py','formal_resume.py','formal_closure.py']:
            with (payload/'trial-code'/name).open('xb') as f:f.write((root/name).read_bytes())
        state['input_verified']=True;state['source_checkpoint_sha256']=CHECKPOINT_SHA;state['stage']='install';emit('stage',stage='install')
        vendor='/isaac-sim/python.sh';site=work/'dependency-site'
        assert Path(vendor).is_file()
        rc=stage([vendor,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir',
                  '--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(payload/'deps/wheels'),
                  '--target',str(site),'-r',str(payload/'deps/overlay.lock')],work/'install.log',env,start+510)
        if rc:raise ValueError('Offline installation failed')
        env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','');state['stage']='gpu_preflight';emit('stage',stage='gpu_preflight')
        probe="import torch,numpy,json;p=torch.cuda.get_device_properties(0);print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,gpu=p.name,bytes=p.total_memory)),flush=True);assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4' and torch.cuda.device_count()==1 and '4090' in p.name and [p.major,p.minor]==[8,9] and 23<=p.total_memory/2**30<=25"
        if stage([vendor,'-c',probe],work/'gpu-preflight.log',env,start+550):raise ValueError('GPU contract failed')
        if time.monotonic()>start+650:raise TimeoutError('Insufficient training budget')
        out=work/'native-output';out.mkdir();env['PHP_FLUX_TASK_ID']=a.task_id
        state['stage']='native_training';state['native_training_started']=True;emit('stage',stage='native_training',requested_num_envs=4096)
        cmd=[vendor,str(payload/'trial-code/formal_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--thread-id',a.thread_id,'--turn-id',a.turn_id]
        rc=stage(cmd,out/'native.log',env,start+1710,stop_file=out/'STOP_AFTER_UPDATE',soft=start+1650)
        state['native_exit_code']=rc
        if rc:raise ValueError('Native process nonzero exit')
        state['stage']='result_validation'
        from formal_closure import validate_worker_result
        state.update(validate_worker_result(out,rc,a.task_id));state.update(worker_passed=True,stage='complete')
    except Exception as e:
        state['error_type']=type(e).__name__;emit('stage_failure',stage=state['stage'],error_type=type(e).__name__)
    state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3)
    result=pack(work,state)
    receipt=transfer(caps['output'],result,method='PUT',size=result.stat().st_size,sha256=digest(result),absolute_deadline=start+1790)
    emit('result_uploaded',task_id=a.task_id,bytes=receipt['bytes'],sha256=receipt['sha256'],worker_passed=state['worker_passed'],elapsed_seconds=round(time.monotonic()-start,3))
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:emit('bootstrap_failure',error_type=type(e).__name__);sys.exit(1)