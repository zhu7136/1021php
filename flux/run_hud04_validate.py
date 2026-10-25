"""Single bounded HuD04 integration attempt:900s total worker, no retries.

Only this generic entry is public. Inputs arrive through a hashed scoped GCS
read; results leave through a separate scoped GCS write. No account credentials.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
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


def fetch(url,path,size,sha):
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.hostname not in ('storage.googleapis.com','files.pythonhosted.org') or u.username or u.password:
        raise ValueError('Unapproved input service')
    if not 0<size<400*1024*1024: raise ValueError('Input outside size bound')
    opener=urllib.request.build_opener(NoRedirect())
    with opener.open(url,timeout=45) as incoming,path.open('xb') as out:
        remaining=size
        while remaining:
            block=incoming.read(min(1024*1024,remaining))
            if not block: raise ValueError('Incomplete input')
            out.write(block);remaining-=len(block)
        if incoming.read(1): raise ValueError('Oversized input')
    if support.digest(path)!=sha: raise ValueError('Input digest mismatch')


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


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-b64',required=True)
    args=parser.parse_args()
    control=support.decode_control(args.control_b64)
    assert control['worker_seconds']==900 and control['attempt'] in (1,2)
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
        stage='scoped_input_delivery'
        payload=out/'input.zip'
        fetch(control['input_url'],payload,control['input_bytes'],control['input_sha256'])
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
        support.run([str(python),str(ROOT/'flux/hud04_validate_child.py')],env,remaining,out/'integration.log')
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