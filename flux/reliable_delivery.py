"""Credential-free, fail-closed signed-GCS delivery with explicit verification levels.

No torch/simulator imports, no API keys, no redirects, no file deletion.
PUT success alone is never delivery success. Real worker egress is unproven
until preflight runs there. Capabilities must be supplied by the controller.
"""
from __future__ import annotations
import base64,hashlib,json,os,socket,ssl,time,urllib.error,urllib.parse,urllib.request,zipfile
from pathlib import Path

MAX_BYTES=32*1024*1024
ARMS=tuple(f'{label}-seed{seed}' for seed in (1402,2402,3402) for label in ('before','after'))
FILES=dict(probe='delivery_probe.json',checkpoint='resume_trained_checkpoint.pt',
    training_summary='resume_training_summary.json',summary='resume_summary.json',bundle='protected_resume_bundle.zip',
    **{arm:arm+'_evaluation.json' for arm in ARMS})

class DeliveryError(RuntimeError):
    def __init__(self,diagnostic):
        self.diagnostic=diagnostic
        super().__init__('Output delivery failed: '+json.dumps(diagnostic,sort_keys=True))

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def error_details(exc):
    """Do not emit raw exception messages: they may include signed capabilities."""
    if isinstance(exc,DeliveryError):return dict(exc.diagnostic)
    result=dict(exception_type=type(exc).__name__)
    reason=exc.reason if isinstance(exc,urllib.error.URLError) else exc
    result['reason_type']=type(reason).__name__
    if isinstance(exc,urllib.error.HTTPError):result['http_status']=exc.code
    if isinstance(reason,OSError) and isinstance(reason.errno,int):result['errno']=reason.errno
    if isinstance(reason,socket.gaierror):result['category']='dns'
    elif isinstance(reason,ssl.SSLError):result['category']='tls'
    elif isinstance(reason,(TimeoutError,socket.timeout)):result['category']='timeout'
    elif isinstance(exc,urllib.error.HTTPError):result['category']='http'
    elif isinstance(reason,OSError):result['category']='connection'
    else:result['category']='other'
    return result

def transient(exc):
    if isinstance(exc,urllib.error.HTTPError):return exc.code in (500,502,503,504)
    reason=exc.reason if isinstance(exc,urllib.error.URLError) else exc
    return isinstance(reason,(TimeoutError,ConnectionError)) or (
        isinstance(reason,socket.gaierror) and reason.errno==socket.EAI_AGAIN)

def target(url,run_path,filename):
    p=urllib.parse.urlsplit(url)
    if (p.scheme!='https' or p.hostname!='storage.googleapis.com' or p.port not in (None,443)
        or p.username or p.password or p.fragment):raise ValueError('Only signed GCS HTTPS capabilities allowed')
    expected='/wandb-production.appspot.com/'+run_path+'/'+filename
    if urllib.parse.unquote(p.path)!=expected:raise ValueError('Capability object differs from declared run/output')
    q=urllib.parse.parse_qs(p.query)
    if not (q.get('X-Goog-Signature') or q.get('Signature')):raise ValueError('Signed capability required')
    return p.path

class Delivery:
    def __init__(self,manifest,root,emit,*,open_url=None,sleep=time.sleep):
        if set(manifest)!={'schema_version','run_path','outputs'} or manifest['schema_version']!=1:
            raise ValueError('Unknown delivery manifest schema')
        self.run_path=manifest['run_path']
        parts=self.run_path.split('/')
        if len(parts)!=3 or any(not p or p in ('.','..') for p in parts):raise ValueError('Exact run path required')
        if set(manifest['outputs'])!=set(FILES):raise ValueError('All independent output capabilities required')
        self.outputs=manifest['outputs'];self.root=Path(root);self.emit=emit
        self.open_url=open_url or urllib.request.build_opener(NoRedirect()).open
        self.sleep=sleep;self.preflight_verified=False;self.receipts=[]
        for key,entry in self.outputs.items():
            if set(entry)!={'put_url','get_url','headers'}:raise ValueError('PUT and GET capabilities required')
            if entry['put_url']==entry['get_url']:raise ValueError('Distinct method-scoped capabilities required')
            target(entry['put_url'],self.run_path,FILES[key])
            if entry['get_url'] is not None:target(entry['get_url'],self.run_path,FILES[key])
            elif key=='probe':raise ValueError('Probe requires signed GET capability before GPU work')
            if not isinstance(entry['headers'],dict) or any(
                h.lower() not in ('content-type','x-goog-content-sha256','x-goog-if-generation-match','x-goog-hash')
                or not isinstance(v,str) or '\n' in v or '\r' in v for h,v in entry['headers'].items()):
                raise ValueError('API credentials or unsafe upload headers forbidden')

    def _request(self,request):
        with self.open_url(request,timeout=20) as response:
            if response.status not in (200,201):
                raise urllib.error.HTTPError(request.full_url,response.status,'HTTP rejected',{},None)
            if request.get_method()=='PUT':return dict(response.headers)
            # Bound even a misleading Content-Length and avoid reading attacker-sized data.
            chunks=[];total=0
            while True:
                chunk=response.read(min(1024*1024,MAX_BYTES+1-total))
                if not chunk:break
                total+=len(chunk)
                if total>MAX_BYTES:raise ValueError('Remote object exceeds bound')
                chunks.append(chunk)
            return b''.join(chunks)

    def _put_get(self,key,data):
        if len(data)>MAX_BYTES:raise ValueError('Output exceeds byte bound')
        entry=self.outputs[key];digest=hashlib.sha256(data).hexdigest()
        # Only explicitly transient failures retried once. Same bytes/object; never retry training.
        for attempt in (1,2):
            phase='put'
            try:
                raw_headers=self._request(urllib.request.Request(entry['put_url'],data=data,headers=entry['headers'],method='PUT'))
                headers={k.lower():v for k,v in raw_headers.items()}
                md5=hashlib.md5(data).digest()
                hashes=[h.strip() for h in headers.get('x-goog-hash','').split(',')]
                storage_verified=('md5='+base64.b64encode(md5).decode()) in hashes or headers.get('etag','').strip('"')==md5.hex()
                if entry['get_url'] is not None:
                    phase='get'
                    remote=self._request(urllib.request.Request(entry['get_url'],method='GET'))
                    if len(remote)!=len(data) or hashlib.sha256(remote).hexdigest()!=digest:
                        raise DeliveryError(dict(category='hash_mismatch',phase='get',output=key,attempt=attempt))
                    download_verified=True
                else:
                    if not storage_verified:raise DeliveryError(dict(category='missing_or_mismatched_storage_md5',phase='put',output=key,attempt=attempt))
                    download_verified=False
                receipt=dict(output=key,filename=FILES[key],bytes=len(data),sha256=digest,
                    put_succeeded=True,storage_md5_verified=storage_verified,
                    download_sha256_verified=download_verified,controller_download_required=not download_verified,attempt=attempt)
                self.receipts.append(receipt)
                self.emit('output_download_verified' if download_verified else 'output_stored_pending_controller_download',**receipt)
                return receipt
            except Exception as exc:
                diagnostic={**error_details(exc),'output':key,'phase':phase,'attempt':attempt}
                retry=attempt==1 and transient(exc)
                self.emit('output_delivery_failed',**diagnostic,will_retry=retry)
                if not retry:raise DeliveryError(diagnostic) from None
                self.sleep(1)

    def _record(self):
        self.root.mkdir(parents=True,exist_ok=True)
        temp=self.root/'delivery_receipts.json.partial'
        temp.write_text(json.dumps(dict(preflight_verified=self.preflight_verified,receipts=self.receipts),indent=2))
        temp.replace(self.root/'delivery_receipts.json')

    def preflight(self):
        data=json.dumps(dict(kind='output_delivery_preflight',run_path=self.run_path),sort_keys=True).encode()
        result=self._put_get('probe',data)
        self.preflight_verified=True;self._record()
        return result

    def send_file(self,key,path):
        if not self.preflight_verified:raise RuntimeError('Delivery preflight must precede GPU work')
        path=Path(path)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Output must be a regular file under the local output root')
        if path.stat().st_size>MAX_BYTES:raise ValueError('Output exceeds byte bound')
        result=self._put_get(key,path.read_bytes());self._record();return result

    def stage_complete(self,arm,directory):
        directory=Path(directory)
        if arm=='training':
            # Protect the new model BEFORE paying for evaluations or compressing diagnostics.
            self.send_file('checkpoint',directory/'model_final.pt')
            self.send_file('training_summary',directory/'summary.json')
        elif arm in ARMS:self.send_file(arm,directory/'evaluation.json')
        else:raise ValueError('Unknown delivery stage')

    def finish(self):
        if not self.preflight_verified:
            self.emit('output_delivery_skipped',reason='preflight_failed_no_gpu_work_authorized_by_delivery_gate')
            return
        if (self.root/'summary.json').is_file():self.send_file('summary',self.root/'summary.json')
        files=[p for p in self.root.glob('*.json') if p.is_file()]
        for name in ('training',)+ARMS:
            files.extend(p for p in (self.root/name).glob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.yaml'))
        if any(p.is_symlink() or not p.resolve().is_relative_to(self.root.resolve()) for p in files):raise ValueError('Unsafe output')
        if sum(p.stat().st_size for p in files)>128*1024*1024:raise ValueError('Raw output exceeds byte bound')
        archive=self.root/'protected_resume_bundle.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files):z.write(p,arcname=str(p.relative_to(self.root)))
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('ZIP integrity failure')
        return self.send_file('bundle',archive)