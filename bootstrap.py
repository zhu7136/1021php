"""One Hu-D04 trial, <=600s total worker, no account keys, no retry."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zipfile
from bootstrap_base import safe_extract,clean_env
from scoped_transport import transfer,digest,validate_capability
from transfer_diagnostics import download

ACTIVE=set()

def stage(command,log,env,hard,stop_file=None,soft=None):
    if time.monotonic()>=hard:raise TimeoutError('No stage budget')
    with Path(log).open('xb') as f:
        child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        ACTIVE.add(child.pid)
        try:
            while child.poll() is None:
                now=time.monotonic()
                if stop_file and now>=soft:Path(stop_file).touch(exist_ok=True)
                if now>=hard:raise TimeoutError('Stage deadline')
                time.sleep(.1)
            return child.returncode
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=2)
            ACTIVE.discard(child.pid)

def watchdog(start):
    time.sleep(max(0,start+598-time.monotonic()))
    for pid in list(ACTIVE):
        try:os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    os._exit(124)

def pack(work,state):
    (work/'bootstrap-status.json').write_text(json.dumps(state,indent=2))
    rows=[];skipped=[];total=0
    for p in sorted(work.rglob('*')):
        rel=p.relative_to(work)
        if p.is_symlink():skipped.append(str(rel));continue
        if not p.is_file() or rel.parts[0] in {'payload','dependency-site'}:continue
        if p.name in {'input.zip','flux-result.zip'}:continue
        if p.suffix=='.pt' and p.name!='model_smoke_final.pt':continue
        if p.suffix not in {'.pt','.json','.jsonl','.log','.yaml','.yml'} and 'tfevents' not in p.name:continue
        total+=p.stat().st_size
        if total>160*1024**2:raise ValueError('Result size exceeded')
        rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':digest(p)})
    manifest={'task_id':state['task_id'],'source_commit':state['bootstrap_commit'],'files':rows,
        'skipped_symlinks':skipped,'automatic_retries':0,'worker_seconds_max':600}
    archive=work/'flux-result.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('result-manifest.json',json.dumps(manifest,indent=2))
        for r in rows:z.write(work/r['path'],r['path'])
    return archive

def main():
    start=time.monotonic()
    p=argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true')
    for n in ['task-id','commit','input-cap','output-cap','input-size','input-sha256','thread-id','turn-id']:
        p.add_argument('--'+n)
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({'execution_requested':False,'worker_seconds_max':600,'automatic_retries':0}));return 0
    if not all(vars(a).values()):raise ValueError('Missing bound scope')
    threading.Thread(target=watchdog,args=(start,),daemon=True).start()
    capin=json.loads(base64.urlsafe_b64decode(a.input_cap));capout=json.loads(base64.urlsafe_b64decode(a.output_cap))
    validate_capability(capin,'GET');validate_capability(capout,'PUT')
    env=clean_env();root=Path(__file__).resolve().parent
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=env,text=True,timeout=5).strip()
    if head!=a.commit:raise ValueError('Unpinned source')
    work=root/('private-'+a.task_id);work.mkdir(exist_ok=False)
    state={'task_id':a.task_id,'bootstrap_commit':head,'worker_seconds_max':600,'automatic_retries':0,
        'worker_passed':False,'teacher_converged':False,'native_training_started':False,'stage':'input'}
    try:
        print(json.dumps({'event':'stage','stage':'input','task_id':a.task_id}),flush=True)
        archive=work/'input.zip'
        state['input_transfer']={}
        download(capin,archive,size=int(a.input_size),sha256=a.input_sha256,
            absolute_deadline=start+120,progress=state['input_transfer'])
        manifest=safe_extract(archive,work/'payload');payload=work/'payload'
        assert manifest['scope']['worker_seconds_max']==600 and manifest['scope']['num_envs']==128
        state['input_verified']=True;state['stage']='install'
        print(json.dumps({'event':'stage','stage':'install'}),flush=True)
        vendor='/isaac-sim/python.sh';site=work/'dependency-site'
        if not Path(vendor).is_file():raise ValueError('Vendor Python missing')
        rc=stage([vendor,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir',
            '--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(payload/'deps/wheels'),
            '--target',str(site),'-r',str(payload/'deps/overlay.lock')],work/'install.log',env,start+210)
        if rc:raise ValueError('Offline dependency installation failed')
        env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','')
        state['stage']='gpu_preflight'
        probe="import torch,numpy,json; p=torch.cuda.get_device_properties(0);print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,gpu=p.name,bytes=p.total_memory)),flush=True);assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4' and torch.cuda.device_count()==1 and '4090' in p.name and [p.major,p.minor]==[8,9] and 23<=p.total_memory/2**30<=25"
        if stage([vendor,'-c',probe],work/'gpu-preflight.log',env,start+250):raise ValueError('GPU contract failed')
        if time.monotonic()>start+330:raise TimeoutError('Too little training budget')
        out=work/'native-output';out.mkdir()
        env['PHP_FLUX_TASK_ID']=a.task_id
        state['stage']='native_training';state['native_training_started']=True
        print(json.dumps({'event':'stage','stage':'native_training'}),flush=True)
        cmd=[vendor,str(payload/'trial-code/trial_child.py'),'--payload',str(payload),'--output',str(out),
            '--site',str(site),'--thread-id',a.thread_id,'--turn-id',a.turn_id]
        rc=stage(cmd,out/'native.log',env,start+515,stop_file=out/'STOP_AFTER_UPDATE',soft=start+455)
        state['native_exit_code']=rc
        if rc:raise ValueError('Native child did not exit cleanly')
        state['stage']='result_validation'
        sys.path.insert(0,str(payload/'trial-code'))
        from closure_evidence import validate_worker_result
        evidence=validate_worker_result(out,rc,a.task_id)
        state.update(evidence)
        # worker_passed means bounded learning evidence+zero native exit;
        # simulator-close return remains an independent explicit result field.
        state.update(worker_passed=True,completed_updates=evidence['completed_updates'],stage='complete')
    except Exception as e:
        state['error_type']=type(e).__name__
        print(json.dumps({'event':'stage_failure','stage':state['stage'],'error_type':type(e).__name__}),flush=True)
    state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3)
    result=pack(work,state)
    receipt=transfer(capout,result,method='PUT',size=result.stat().st_size,sha256=digest(result),absolute_deadline=start+590)
    print(json.dumps({'event':'result_uploaded','task_id':a.task_id,'bytes':receipt['bytes'],'sha256':receipt['sha256'],
        'worker_passed':state['worker_passed'],'elapsed_seconds':round(time.monotonic()-start,3)}),flush=True)
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print(json.dumps({'event':'bootstrap_failure','error_type':type(e).__name__}),flush=True)
        sys.exit(1)