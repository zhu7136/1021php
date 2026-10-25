"""Single bounded HuD04 integration attempt:900s total worker, no retries.

Only this generic entry is public. Inputs arrive through a hashed scoped GCS
read; results leave through a separate scoped GCS write. No account credentials.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flux import run_smoke as support


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('Scoped input redirects prohibited')


def fetch(url,path,size,sha,progress=None):
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.hostname not in ('storage.googleapis.com','files.pythonhosted.org') or u.username or u.password:
        raise ValueError('Unapproved input service')
    if not 0<size<400*1024*1024: raise ValueError('Input outside size bound')
    opener=urllib.request.build_opener(NoRedirect())
    with opener.open(url,timeout=45) as incoming,path.open('xb') as out:
        remaining=size
        while remaining:
            # read1 returns available data without waiting to fill a 1MiB read.
            block=incoming.read1(min(64*1024,remaining))
            if not block: raise ValueError('Incomplete input')
            out.write(block);remaining-=len(block)
            if progress is not None: progress(size-remaining)
        if incoming.read(1): raise ValueError('Oversized input')
    if support.digest(path)!=sha: raise ValueError('Input digest mismatch')


def fetch_input(url,path,size,sha,*,max_seconds=120,progress_interval=5):
    """Main-thread POSIX deadline, including open/read/hash; keep worker alarm.

    This protects the paid worker budget, not a claim of worker connectivity.
    No URL, headers or exception messages are persisted in transfer diagnostics.
    """
    if not math.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('Positive finite input delivery budget required')
    if not math.isfinite(progress_interval) or progress_interval<=0:
        raise ValueError('Positive finite progress interval required')
    report=path.parent/'input-transfer.json'
    if path.exists() or report.exists():
        raise FileExistsError('Input delivery must start in a fresh output directory')
    started=time.monotonic()
    previous_handler=signal.getsignal(signal.SIGALRM)
    previous_timer=signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[1]!=0:
        raise ValueError('Periodic worker alarm unsupported')
    earlier_worker=0<previous_timer[0]<=max_seconds
    limit=min(max_seconds,previous_timer[0]) if previous_timer[0]>0 else max_seconds
    state={'phase':'downloading','expected_bytes':size,'bytes_received':0,
           'max_seconds':max_seconds,'effective_deadline_seconds':limit,
           'sha256_verified':False}
    last_progress=started

    def persist():
        elapsed=time.monotonic()-started
        fields={**state,'elapsed_seconds':round(elapsed,3),
                'average_bytes_per_second':round(state['bytes_received']/max(elapsed,1e-9),1)}
        temporary=report.with_suffix('.json.part')
        with temporary.open('w') as stream:
            json.dump(fields,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        temporary.replace(report)
        support.emit('hud04_input_transfer',**fields)

    def progress(count):
        nonlocal last_progress
        state['bytes_received']=count
        now=time.monotonic()
        if now-last_progress>=progress_interval:
            persist();last_progress=now

    def deadline(signum,frame):
        if earlier_worker and callable(previous_handler):
            previous_handler(signum,frame)
        raise TimeoutError('Input delivery total deadline exceeded')

    persist()
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,limit)
    try:
        fetch(url,path,size,sha,progress=progress)
        state.update(phase='verified',sha256_verified=True,bytes_received=size)
    except BaseException as exc:
        state.update(phase='failed',error_type=type(exc).__name__)
        raise
    finally:
        # Never reset the worker to its full original allowance.
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0:
            remaining=previous_timer[0]-(time.monotonic()-started)
            if remaining>0: signal.setitimer(signal.ITIMER_REAL,remaining)
        persist()


def unpack(path,destination):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or len(names)>1200: raise ValueError('Invalid archive')
        if sum(x.file_size for x in archive.infolist())>300*1024*1024: raise ValueError('Expanded size exceeded')
        for info in archive.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or info.is_dir():
                raise ValueError('Unsafe input member')
            if (info.external_attr>>16)&0o170000==0o120000: raise ValueError('No input symlinks')
            target=destination/str(p)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out: out.write(archive.read(info))


def deliver_input(control,payload):
    """Explicit source choice: fail closed, never silently change delivery."""
    if 'input_git_manifest' in control:
        if 'input_url' in control:
            raise ValueError('Ambiguous Git/HTTPS input sources')
        from flux.hud04_git_input import assemble_git_input
        report=assemble_git_input(ROOT,control['input_git_manifest'],
            control['input_git_manifest_sha256'],payload,
            control['input_bytes'],control['input_sha256'])
        (payload.parent/'input-git-verification.json').write_text(json.dumps(report,indent=2))
        support.emit('hud04_git_input_verified',**report)
    else:
        if 'input_git_manifest_sha256' in control:
            raise ValueError('Incomplete Git input source')
        fetch_input(control['input_url'],payload,control['input_bytes'],control['input_sha256'])


def wait_for_integration(child, out, timeout, shutdown_grace=30):
    """Bound post-report shutdown independently; a report alone is not success."""
    end=time.monotonic()+timeout
    shutdown_end=None
    while True:
        reports=(out/'integration-result.json',out/'child-failure.json')
        if shutdown_end is None and any(p.is_file() for p in reports):
            shutdown_end=min(end,time.monotonic()+shutdown_grace)
        bound=end if shutdown_end is None else shutdown_end
        remaining=bound-time.monotonic()
        if remaining<=0:
            reason='shutdown_timeout' if shutdown_end is not None else 'integration_timeout'
            (out/'process-exit.json').write_text(json.dumps({
                'normal_exit_verified':False,'reason':reason,
                'shutdown_grace_seconds':shutdown_grace}))
            raise TimeoutError(reason)
        try:
            code=child.wait(timeout=min(.5,remaining))
        except subprocess.TimeoutExpired:
            continue
        complete=out/'shutdown-complete.json'
        closed=False
        if complete.is_file():
            closed=json.loads(complete.read_text()).get('app_close_returned') is True
        verified=code==0 and closed and not (out/'child-failure.json').exists()
        (out/'process-exit.json').write_text(json.dumps({
            'normal_exit_verified':verified,'returncode':code,'app_close_returned':closed}))
        if not verified:
            raise RuntimeError('Child failed or simulator shutdown was not verified')
        return


def run_integration(command, env, timeout, out):
    with (out/'integration.log').open('xb') as log:
        support.CHILD=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,
                                       stdout=log,stderr=subprocess.STDOUT)
        try:
            wait_for_integration(support.CHILD,out,timeout)
        finally:
            support.stop_child()


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-b64',required=True)
    args=parser.parse_args()
    control=support.decode_control(args.control_b64)
    assert control['worker_seconds']==900 and control['attempt'] in (1,2,3,4,5)
    out=support.task_output(control['task_id'])
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for name in ('WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY','GITHUB_TOKEN','github','key'):
        env.pop(name,None)
    env.update(PYTHONUNBUFFERED='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0')
    def deadline(*_):
        support.stop_child()
        raise TimeoutError('HuD04 worker bound reached')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(840)
    stage='source_verification';passed=False
    try:
        from flux.preflight import static_checks
        static_checks(expected_commit=control['commit'])
        python=support.launcher()
        probe=out/'worker-probe.json'
        probe.write_text(json.dumps({'task_id':control['task_id'],'training_started':False}))
        support.upload(control['probe_url'],control['headers'],probe)
        support.emit('hud04_probe_begin',worker_seconds=900,automatic_retries=0)
        stage='git_input_delivery' if 'input_git_manifest' in control else 'scoped_input_delivery'
        payload=out/'input.zip'
        deliver_input(control,payload)
        capsule=out/'capsule';capsule.mkdir()
        unpack(payload,capsule)
        # Copy only the explicit HuD04 Python overlay; do not modify G1 files.
        overlay=capsule/'overlay'
        for src in overlay.rglob('*.py'):
            rel=src.relative_to(overlay)
            if rel.parts[0] not in ('wbt_training','flux'): raise ValueError('Unexpected overlay root')
            target=ROOT/rel
            if target.exists(): raise ValueError('Refusing existing source overwrite')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as dest: dest.write(src.read_bytes())
        stage='pinned_dependency_delivery'
        private=out/'upstream-wheels';private.mkdir()
        rows=json.loads((ROOT/'flux/smoke-upstream-wheels.json').read_text())
        assert len(rows)==8
        def wheel(row):
            if Path(row['file']).name!=row['file']: raise ValueError('Invalid wheel name')
            fetch(row['url'],private/row['file'],row['bytes'],row['sha256'])
        with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(wheel,rows))
        stage='offline_dependency_install'
        inputs=out/'inputs'
        support.run([str(python),str(ROOT/'flux/prepare_inputs.py'),'--output',str(inputs),
                     '--install','--python',str(python),'--withheld-wheel-dir',str(private)],
                     env,min(240,max(1,int(800-(time.monotonic()-started)))),out/'prepare.log')
        env['PYTHONPATH']=os.pathsep.join([str(inputs/'.flux-deps'),str(ROOT),
            str(ROOT/'thirdparty/holosoma/src/holosoma'),env.get('PYTHONPATH','')])
        env.update(HUD04_VALIDATE_OUTPUT=str(out),HUD04_VALIDATE_INPUT=str(capsule/'input'))
        stage='real_simulator_integration'
        remaining=int(810-(time.monotonic()-started))
        if remaining<90: raise TimeoutError('No integration budget remains')
        support.emit('hud04_real_simulator_start',remaining_seconds=remaining)
        run_integration([str(python),str(ROOT/'flux/hud04_validate_child.py')],env,remaining,out)
        if (out/'child-failure.json').exists():
            stage='child_failure_preserved'
            raise RuntimeError('Child recorded an integration exception before simulator teardown')
        if not (out/'integration-result.json').is_file():
            stage='child_missing_acceptance_report'
            raise RuntimeError('Child exited without an acceptance report; integration is unverified')
        result=json.loads((out/'integration-result.json').read_text())
        assert result['integration_passed'] is True
        passed=True
    except BaseException as exc:
        # No scoped URL/argv/exception message is written to public console.
        (out/'failure.json').write_text(json.dumps({'stage':stage,'type':type(exc).__name__,
            'frames':[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(exc.__traceback__)]}))
        support.emit('hud04_probe_failed',stage=stage,type=type(exc).__name__)
    finally:
        support.stop_child()
        signal.alarm(max(1,int(898-(time.monotonic()-started))))
        members=[p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.log')]
        manifest={'passed':passed,'training_started':False,'elapsed_seconds':time.monotonic()-started,
                  'files':[{'path':p.name,'sha256':support.digest(p),'bytes':p.stat().st_size} for p in members]}
        (out/'result-manifest.json').write_text(json.dumps(manifest,indent=2))
        resultzip=out/'hud04-validation-result.zip'
        with zipfile.ZipFile(resultzip,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in members+[out/'result-manifest.json']: archive.write(path,path.name)
        support.upload(control['output_url'],control['headers'],resultzip)
        support.emit('hud04_result_uploaded',passed=passed,elapsed_seconds=round(time.monotonic()-started,1))
        signal.alarm(0)
    return 0 if passed else 1


if __name__=='__main__': sys.exit(main())