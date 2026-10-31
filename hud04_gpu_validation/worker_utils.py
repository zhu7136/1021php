"""Local worker helpers, with no dispatch, credentials, or import-time actions."""
import json,os,signal,subprocess,time,zipfile
from pathlib import Path
from scoped_transport import digest
ACTIVE=set()

def emit(event,**fields):print(json.dumps({'event':event,**fields}),flush=True)

def stage(command,log,env,hard,stop_file=None,soft=None):
    if time.monotonic()>=hard:raise TimeoutError('No stage budget')
    pos=0;last=0
    with Path(log).open('xb') as f:
        child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        ACTIVE.add(child.pid)
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
                        safe={k:v for k,v in obj.items() if isinstance(v,(int,float,bool))}
                        emit('native_progress',native_event=obj['event'],**safe)
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
        if p.is_symlink() or not p.is_file() or rel.parts[0] in {'payload','dependency-site','overlay'}:continue
        if p.name in {'input.zip','source-result.zip','flux-result.zip','full-input.zip'}:continue
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