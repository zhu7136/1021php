"""HuD04 r18: one fresh task, signed single-object input/output, native r15 train.

No credentials or ticket values are logged. No task creation occurs here.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import zipfile
from scoped_transport import deadline,digest,transfer,validate_capability

MAX_EXPANDED=512*1024*1024


def fallback_reports(work):
    """Bounded regular JSON evidence if the inner seal failed. Never follow links."""
    work=Path(work).resolve(strict=True)
    names=('process-state.json','progress.json','native_reload.json','learning_state.json',
        'resume-restored.json','resume-continued.json','train-returned.json','runtime-versions.json','run-metadata.json',
        'close-entered.json','close-settings.json','close-returned.json','close-error.json','close-stack.log',
        'lifecycle-events.jsonl','lifecycle-returned.json','helper-before.json','helper-at-close.json',
        'helper-after.json','helper-variant.json','robot-probe.json','close-calls.jsonl',
        'physics-cleanup-entered.json','physics-cleanup-returned.json',
        'physics-cleanup-error.json','physics-cleanup-stack.log','child-error.json')
    files=[];total=0
    for name in names:
        p=work/'native-output'/name
        if p.is_symlink() or any(parent.is_symlink() for parent in p.parents):continue
        if not p.is_file() or not p.resolve().is_relative_to(work):continue
        size=p.stat().st_size
        if size>256*1024 or total+size>1024*1024:continue
        total+=size;files.append(p)
    return files


def safe_extract(archive,destination):
    destination=Path(destination)
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        rows=z.infolist()
        if len(rows)>1100 or sum(i.file_size for i in rows)>MAX_EXPANDED:raise ValueError('Input archive limit')
        seen=set()
        for i in rows:
            p=PurePosixPath(i.filename)
            mode=i.external_attr>>16
            if (not i.filename or p.is_absolute() or '..' in p.parts or '\\' in i.filename
                or i.filename in seen or i.is_dir() or stat.S_ISLNK(mode)
                or stat.S_IFMT(mode) not in (0,stat.S_IFREG)):
                raise ValueError('Nonregular/duplicate/escaping archive member')
            seen.add(i.filename)
            out=destination.joinpath(*p.parts)
            if not out.resolve().is_relative_to(destination.resolve()):raise ValueError('Archive path escape')
            out.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as source,out.open('xb') as target:
                while True:
                    chunk=source.read(1024*1024)
                    if not chunk:break
                    target.write(chunk)
    manifest=json.loads((destination/'payload-manifest.json').read_text())
    expected={'payload-manifest.json'}
    for r in manifest['files']:
        rel=PurePosixPath(r['path'])
        if rel.is_absolute() or '..' in rel.parts or r['path'] in expected:raise ValueError('Invalid payload manifest')
        expected.add(r['path']);f=destination.joinpath(*rel.parts)
        if f.stat().st_size!=r['bytes'] or digest(f)!=r['sha256']:raise ValueError('Payload mismatch')
    if seen!=expected:raise ValueError('Unlisted payload file')
    return manifest


def clean_env():
    # Only runtime/library settings, never inherited user account credentials.
    allowed=('PATH','HOME','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','CONDA_PREFIX',
        'VIRTUAL_ENV','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES',
        'DISPLAY','LANG','LC_ALL','TERM','ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','EXP_PATH',
        'CARB_APP_PATH','OMNI_KIT_ALLOW_ROOT')
    env={k:os.environ[k] for k in allowed if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',WANDB_MODE='offline',
        WANDB_SILENT='true',PIP_CONFIG_FILE=os.devnull)
    return env


def run_bounded(command,log,env,end):
    if end<=time.monotonic():raise TimeoutError('Stage budget exhausted')
    with Path(log).open('xb') as stream:
        p=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if time.monotonic()>=end:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:p.wait(timeout=6)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        p.wait(timeout=5)
                    raise TimeoutError('Bounded subprocess deadline')
                time.sleep(.1)
        finally:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:p.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    p.wait(timeout=5)
    return p.returncode


class SafeParser(argparse.ArgumentParser):
    def error(self,message):raise ValueError('Invalid bootstrap arguments')


def main():
    start=time.monotonic();end=start+900
    p=SafeParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--accept-task-ticket-visibility-risk',action='store_true')
    for n in ('task-id','commit','input-cap','output-cap','input-size','input-sha256','thread-id','turn-id'):
        p.add_argument('--'+n)
    a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(execution_requested=False,gpu_started=False,worker_seconds=900,automatic_retries=0)),flush=True);return 0
    if not all(vars(a).values()) or not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id):raise ValueError('Bound scope required')
    cap_in=json.loads(base64.urlsafe_b64decode(a.input_cap));cap_out=json.loads(base64.urlsafe_b64decode(a.output_cap))
    validate_capability(cap_in,'GET');validate_capability(cap_out,'PUT')
    root=Path(__file__).resolve().parent
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=clean_env(),timeout=10,text=True).strip()
    if head!=a.commit:raise ValueError('Unpinned bootstrap checkout')
    work=root/('private-'+a.task_id)
    work.mkdir(exist_ok=False)
    state=dict(task_id=a.task_id,bootstrap_commit=head,worker_seconds_max=900,automatic_retries=0,
        input_roundtrip_verified=False,install_verified=False,worker_passed=False,
        parameter_visibility_risk_accepted=a.accept_task_ticket_visibility_risk,account_credentials_forwarded=False)
    stage='input_download'
    try:
        with deadline(end-40):
            print(json.dumps(dict(event='hud04_stage',stage=stage,task_id=a.task_id)),flush=True)
            archive=work/'input.zip'
            transfer(cap_in,archive,method='GET',size=int(a.input_size),sha256=a.input_sha256,
                absolute_deadline=min(end-180,time.monotonic()+240))
            stage='input_extract_verify'
            manifest=safe_extract(archive,work/'payload')
            state['input_roundtrip_verified']=True
            print(json.dumps(dict(event='hud04_stage',stage='input_verified',task_id=a.task_id)),flush=True)
            pins=manifest['scope'];payload=work/'payload';source=payload/'native-source'
            if (pins['worker_seconds_max']!=900 or pins['training_updates']!=0 or pins['num_envs']!=2
                or pins['explicit_control_steps']!=8 or pins['checkpoint_loaded'] is not False
                or pins['learner_instantiated'] is not False or pins['physics_cleanup_budget_seconds']!=30
                or pins['native_close_budget_seconds']!=30 or pins['automatic_retries']!=0
                or pins['usd_close_stage_patch'] is not False):
                raise ValueError('Unexpected workload scope')
            env=clean_env();vendor='/isaac-sim/python.sh'
            if not Path(vendor).is_file():raise ValueError('Vendor interpreter missing')
            stage='dependency_install'
            print(json.dumps(dict(event='hud04_stage',stage=stage,task_id=a.task_id)),flush=True)
            if end-time.monotonic()<180:raise TimeoutError('Insufficient installation/learning budget')
            site=work/'dependency-site'
            code=run_bounded([vendor,'-m','pip','install','--isolated','--disable-pip-version-check',
                '--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes',
                '--find-links',str(payload/'deps/wheels'),'--target',str(site),
                '-r',str(payload/'deps/overlay.lock')],work/'install.log',env,min(end-160,time.monotonic()+240))
            state['install_exit_code']=code
            if code:raise ValueError('Isolated dependency installation failed')
            state['install_verified']=True
            env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','')
            stage='dependency_gpu_preflight'
            probe="import sys; sys.path.insert(0,"+repr(str(site))+"); import torch,numpy,onnx,json; p=torch.cuda.get_device_properties(0); d=dict(torch=torch.__version__,numpy=numpy.__version__,onnx=onnx.__version__,gpu=p.name,memory_bytes=p.total_memory,cc=[p.major,p.minor],gpu_count=torch.cuda.device_count()); print(json.dumps(d),flush=True); assert d['torch']=='2.7.0+cu128' and d['numpy']=='1.26.4' and d['onnx']=='1.17.0' and d['gpu_count']==1 and '4090' in p.name and 23<=p.total_memory/2**30<=25 and [p.major,p.minor]==[8,9]"
            code=run_bounded([vendor,'-c',probe],work/'dependency-gpu-preflight.log',env,min(end-130,time.monotonic()+60))
            state['dependency_gpu_preflight_exit_code']=code
            if code:raise ValueError('Target dependencies/GPU contract failed')
            stage='native_worker'
            print(json.dumps(dict(event='hud04_stage',stage=stage,task_id=a.task_id)),flush=True)
            if end-time.monotonic()<150:raise TimeoutError('Insufficient native worker budget')
            cmd=[vendor,str(root/'worker_wrapper.py'),'--outer-deadline',str(end),
                '--native-source',str(source),'--execute','--bundle',str(payload/'bundle'),
                '--manifest-sha256',pins['bundle_manifest_sha256'],'--source-manifest-sha256',pins['source_manifest_sha256'],
                '--config',pins['config'],'--output',str(work/'native-output'),
                '--task-id',a.task_id,'--python',vendor,'--dependency-site',str(site),
                '--thread-id',a.thread_id,'--turn-id',a.turn_id]
            code=run_bounded(cmd,work/'worker-controller.log',env,end-40)
            state['native_worker_exit_code']=code
            inner=work/'native-output/hud04-ppo-result.zip'
            if inner.is_file():
                with zipfile.ZipFile(inner) as z:
                    result=json.loads(z.read('result-manifest.json'))
                state['worker_passed']=code==0 and result['worker_passed'] is True and result['task_id']==a.task_id
            state['last_stage']=stage
    except Exception as e:
        state.update(last_stage=stage,error_type=type(e).__name__,worker_passed=False)
        print(json.dumps(dict(event='hud04_stage_failure',stage=stage,error_type=type(e).__name__,task_id=a.task_id)),flush=True)
    state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3)
    (work/'bootstrap-status.json').write_text(json.dumps(state,indent=2)+'\n')
    result=work/'flux-result.zip'
    with deadline(end-5):
        with zipfile.ZipFile(result,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name in ('bootstrap-status.json','install.log','dependency-gpu-preflight.log','worker-controller.log','native-output/hud04-ppo-result.zip'):
                f=work/name
                if f.is_file():z.write(f,name)
            if not (work/'native-output/hud04-ppo-result.zip').is_file() and (work/'native-output/native.log').is_file():
                z.write(work/'native-output/native.log','native-output/native.log')
            if not (work/'native-output/hud04-ppo-result.zip').is_file():
                for f in fallback_reports(work):z.write(f,f.relative_to(work).as_posix())
        receipt=transfer(cap_out,result,method='PUT',size=result.stat().st_size,sha256=digest(result),absolute_deadline=end)
    print(json.dumps(dict(event='hud04_result_uploaded',task_id=a.task_id,worker_passed=state['worker_passed'],
        bytes=receipt['bytes'],sha256=receipt['sha256'],elapsed_seconds=round(time.monotonic()-start,3),last_stage=state.get('last_stage'))),flush=True)
    return 0 if state['worker_passed'] else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        # Never print exception text/argv; both can contain scoped ticket values.
        print(json.dumps(dict(event='bootstrap_failure',error_type=type(e).__name__,passed=False)),flush=True)
        sys.exit(1)