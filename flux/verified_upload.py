"""Bounded retries of one scoped idempotent PUT, never a task/GPU retry."""
import hashlib,http.client,time,urllib.parse
from pathlib import Path

class UploadError(ValueError):pass

def upload(url,headers,path,on_event=None):
    p=urllib.parse.urlsplit(url)
    if p.scheme!='https' or p.hostname!='storage.googleapis.com' or p.username or p.password or p.port not in (None,443):
        raise UploadError('untrusted_upload_host')
    clean=dict(headers)
    if any(k.lower() in ('authorization','cookie','host','transfer-encoding','content-length') for k in clean):
        raise UploadError('invalid_upload_headers')
    path=Path(path);size=path.stat().st_size
    def digest():
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        return h.hexdigest()
    initial=digest();clean['Content-Length']=str(size)
    started=time.monotonic();deadline=started+180;rows=[]
    emit=on_event or (lambda *a,**kw:None)
    def remaining():
        left=deadline-time.monotonic()
        if left<=0:raise UploadError('upload_deadline_exceeded')
        return left
    for attempt in range(1,4):
        connection=None;code='';retry=False
        try:
            remaining()
            if path.stat().st_size!=size or digest()!=initial:raise UploadError('upload_source_changed')
            connection=http.client.HTTPSConnection(p.hostname,timeout=min(15,remaining()))
            class Body:
                def __init__(self,f):self.f=f
                def read(self,n):
                    timeout=min(15,remaining())
                    if connection.sock is not None:connection.sock.settimeout(timeout)
                    return self.f.read(n)
            with path.open('rb') as f:
                connection.request('PUT',p.path+'?'+p.query,body=Body(f),headers=clean)
                if connection.sock is not None:connection.sock.settimeout(min(15,remaining()))
                response=connection.getresponse();remaining();status=response.status
                if 200<=status<300:
                    response.read(1024);remaining()
                    if path.stat().st_size!=size or digest()!=initial:raise UploadError('upload_source_changed')
                    rows.append({'attempt':attempt,'outcome':'uploaded','http_status':status})
                    report={'attempts':rows,'attempt_limit':3,'deadline_seconds':180,'bytes':size,'sha256':initial,'task_retries':0}
                    emit('scoped_upload_complete',**report);return report
                code='http_'+str(status);retry=status in (429,500,502,503,504)
        except (TimeoutError,ConnectionError,http.client.IncompleteRead,http.client.RemoteDisconnected):
            code='transfer_interrupted';retry=True
        except UploadError:
            raise
        except Exception:
            raise UploadError('nontransient_upload_error') from None
        finally:
            if connection is not None:connection.close()
        rows.append({'attempt':attempt,'outcome':code})
        emit('scoped_upload_attempt_failed',attempt=attempt,outcome=code,will_retry=bool(retry and attempt<3),task_retries=0)
        if not retry:raise UploadError(code) from None
        if attempt==3:raise UploadError('upload_attempts_exhausted') from None
        time.sleep(min(.5*attempt,remaining()))
    raise UploadError('unreachable_upload_state')