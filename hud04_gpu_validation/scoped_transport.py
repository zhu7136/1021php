"""Single-object transport, adapted from historical run_smoke/resume_input.

No account credentials, URL logging, redirects or automatic retries. Callers
provide capabilities in memory; this module does not establish Flux privacy.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path, PurePosixPath
import re
import signal
import time
from urllib.parse import urlsplit, parse_qs
import zipfile

MAX_BYTES=512*1024*1024


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


@contextmanager
def deadline(absolute):
    """Never restart an enclosing worker's remaining budget."""
    if not math.isfinite(absolute) or absolute<=time.monotonic():
        raise TimeoutError('No transfer budget remains')
    old_handler=signal.getsignal(signal.SIGALRM)
    old_timer=signal.getitimer(signal.ITIMER_REAL)
    if old_timer[1]:raise ValueError('Periodic alarm not supported')
    entered=time.monotonic()
    seconds=absolute-entered
    if old_timer[0]>0:seconds=min(seconds,old_timer[0])
    def alarm(*_):raise TimeoutError('Transfer absolute deadline reached')
    signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,max(.001,seconds))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0:
            remaining=old_timer[0]-(time.monotonic()-entered)
            if remaining>0:signal.setitimer(signal.ITIMER_REAL,remaining)
            elif callable(old_handler):old_handler(signal.SIGALRM,None)
            else:raise TimeoutError('Enclosing deadline reached')


def validate_capability(cap,method):
    if set(cap)!={'url','headers','object_path','method'} or cap['method']!=method:
        raise ValueError('Unexpected capability fields/method')
    u=urlsplit(cap['url'])
    if (u.scheme!='https' or u.hostname!='storage.googleapis.com' or u.port not in (None,443)
            or u.username or u.password or u.fragment or not u.query
            or u.path!=cap['object_path'] or not u.path.startswith('/') or u.path=='/'):
        raise ValueError('Expected one pinned signed GCS object')
    q=parse_qs(u.query,keep_blank_values=True)
    if any(len(v)!=1 for v in q.values()):raise ValueError('Duplicate capability parameter')
    if 'Expires' in q and 'Signature' in q:
        expires=float(q['Expires'][0])
    elif {'X-Goog-Date','X-Goog-Expires','X-Goog-Signature'}<=set(q):
        start=datetime.strptime(q['X-Goog-Date'][0],'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
        expires=start+float(q['X-Goog-Expires'][0])
    else:raise ValueError('Unrecognized signed capability expiry')
    if not math.isfinite(expires) or expires<=time.time():raise ValueError('Expired capability')
    headers=cap['headers']
    if not isinstance(headers,dict) or len(headers)>12:raise ValueError('Invalid capability headers')
    for k,v in headers.items():
        if not isinstance(k,str) or not isinstance(v,str) or '\r' in k+v or '\n' in k+v:
            raise ValueError('Invalid header')
        if k.lower() not in {'content-type','content-md5','x-goog-content-sha256'}:
            raise ValueError('Unexpected header; no account credentials allowed')
    return u,expires


def transfer(cap,path,*,method,size,sha256,absolute_deadline):
    """One actual PUT or GET. A failed partial file is never reused."""
    # Include validation and source hashing in the same absolute budget.
    with deadline(absolute_deadline):
        return _transfer(cap,path,method=method,size=size,sha256=sha256,
            absolute_deadline=absolute_deadline)


def _transfer(cap,path,*,method,size,sha256,absolute_deadline):
    u,expiry=validate_capability(cap,method)
    if type(size) is not int or not 0<size<=MAX_BYTES or not re.fullmatch('[0-9a-f]{64}',sha256):
        raise ValueError('Invalid size/hash bound')
    path=Path(path)
    if path.is_symlink():raise ValueError('No symlink transfer')
    if method=='PUT':
        if not path.is_file() or path.stat().st_size!=size or digest(path)!=sha256:
            raise ValueError('Upload source size/hash mismatch')
    elif method=='GET':
        if path.exists():raise FileExistsError('Fresh download required')
    else:raise ValueError('Unsupported transfer method')
    started=time.monotonic()
    # The request cannot consume time beyond the signed capability lifetime.
    end=min(absolute_deadline,started+max(0,expiry-time.time()))
    headers=dict(cap['headers'])
    count=0
    with deadline(end):
        conn=http.client.HTTPSConnection(u.hostname,timeout=max(.001,min(30,end-time.monotonic())))
        try:
            resource=u.path+'?'+u.query
            if method=='PUT':
                headers['Content-Length']=str(size)
                with path.open('rb') as f:conn.request('PUT',resource,body=f,headers=headers)
            else:conn.request('GET',resource,headers=headers)
            response=conn.getresponse()
            if not 200<=response.status<300:raise RuntimeError('Object transfer HTTP failure')
            if method=='GET':
                declared=response.getheader('Content-Length')
                if declared is not None and int(declared)!=size:raise ValueError('Response length mismatch')
                with path.open('xb') as f:
                    while count<size:
                        b=response.read(min(65536,size-count))
                        if not b:raise ValueError('Incomplete download')
                        f.write(b);count+=len(b)
                    if response.read(1):raise ValueError('Oversized download')
                if digest(path)!=sha256:raise ValueError('Downloaded SHA256 mismatch')
            else:
                response.read(1024);count=size
        finally:conn.close()
    return {'method':method,'bytes':count,'sha256':sha256,'verified':True,
        'elapsed_seconds':round(time.monotonic()-started,3),'automatic_retries':0}


def safe_unpack(archive,destination):
    """No pickle loading or ZIP path escapes; verify manifest after extraction."""
    dest=Path(destination)
    if dest.exists() or any(p.is_symlink() for p in [dest,*dest.parents]):
        raise ValueError('Fresh regular destination required')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(infos)>2000 or len(names)!=len(set(names)) or sum(i.file_size for i in infos)>1024**3:
            raise ValueError('ZIP inventory/expansion bound')
        for info in infos:
            p=PurePosixPath(info.filename)
            if (not p.parts or p.is_absolute() or any(x in ('','.','..') for x in info.filename.split('/'))
                    or '\\' in info.filename or info.is_dir() or (info.external_attr>>16)&0o170000==0o120000):
                raise ValueError('Nonregular or unsafe ZIP member')
        dest.mkdir(parents=True,exist_ok=False)
        for info in infos:
            target=dest/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as incoming,target.open('xb') as out:
                while b:=incoming.read(1024*1024):out.write(b)
    return len(infos)


def verify_r15_result(archive,destination,*,task_id,bundle_pin,archive_sha256):
    if digest(archive)!=archive_sha256:raise ValueError('Result ZIP pin mismatch')
    count=safe_unpack(archive,destination);root=Path(destination)
    manifest=json.loads((root/'result-manifest.json').read_text())
    if (manifest['task_id']!=task_id or manifest['bundle_manifest_sha256']!=bundle_pin
            or manifest['automatic_retries']!=0):raise ValueError('Result binding mismatch')
    rows=manifest['files'];seen=set()
    if len(rows)!=count-1:raise ValueError('Result inventory mismatch')
    for row in rows:
        name=row['path'];p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or name in seen:
            raise ValueError('Invalid result member')
        seen.add(name);f=root/name
        if not f.is_file() or f.stat().st_size!=row['bytes'] or digest(f)!=row['sha256']:
            raise ValueError('Result member hash mismatch')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=seen|{'result-manifest.json'}:raise ValueError('Unexpected result file')
    # Use the exact r15 learning acceptance predicate, without unpickling .pt.
    from worker import accepted
    passed=accepted(root,manifest['process'])
    if manifest['worker_passed'] is not passed:raise ValueError('Worker acceptance mismatch')
    return {'inventory_verified':True,'worker_passed':passed,'files':len(rows),
        'checkpoint_deserialized':False,'native_app_close_independently_verified':False}