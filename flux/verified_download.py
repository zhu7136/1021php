"""Bounded scoped GETs; incomplete bytes never become a consumable input.

At most two TRANSFER retries (not GPU/task retries), 180s shared deadline.
HTTP auth/not-found, wrong host, oversize and hash mismatch fail immediately.
Failed partials are retained for diagnosis; existing files are never replaced.
"""
import hashlib,http.client,json,os,re,socket,time,urllib.error,urllib.parse,urllib.request,uuid
from pathlib import Path

class DownloadError(ValueError):pass
class _Transient(Exception):pass

def check_url(url):
    p=urllib.parse.urlsplit(url)
    if p.scheme!='https' or p.hostname!='storage.googleapis.com' or p.username or p.password or p.port not in (None,443):
        raise DownloadError('untrusted_download_host')

class _ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def open_url(url,timeout):
    req=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
    return urllib.request.build_opener(_ScopedRedirect()).open(req,timeout=timeout)

def download(url,destination,expected_bytes,expected_sha256,diagnostics):
    check_url(url)
    if type(expected_bytes) is not int or not 1<=expected_bytes<=60*1024**2 or not re.fullmatch('[0-9a-f]{64}',expected_sha256):
        raise DownloadError('invalid_download_contract')
    destination=Path(destination);diagnostics=Path(diagnostics)
    if destination.exists() or diagnostics.exists():raise DownloadError('download_path_already_exists')
    started=time.monotonic();deadline=started+180;rows=[]
    def remaining():
        left=deadline-time.monotonic()
        if left<=0:raise DownloadError('download_deadline_exceeded')
        return left
    def record():
        diagnostics.write_text(json.dumps({'attempts':rows,'attempt_limit':3,'deadline_seconds':180,
            'elapsed_seconds':round(time.monotonic()-started,3),'expected_bytes':expected_bytes,
            'expected_sha256':expected_sha256,'task_retries':0},indent=2))
    for attempt in range(1,4):
        row={'attempt':attempt,'bytes_received':0,'outcome':'started'};rows.append(row)
        part=destination.with_name(destination.name+'.attempt'+str(attempt)+'-'+uuid.uuid4().hex[:8]+'.part')
        try:
            remaining();h=hashlib.sha256()
            with open_url(url,timeout=min(15,remaining())) as response,part.open('xb') as out:
                check_url(response.url)
                if response.status!=200:raise DownloadError('unexpected_download_status')
                length=response.headers.get('Content-Length')
                if length is not None:
                    if not length.isdigit():raise DownloadError('invalid_content_length')
                    if int(length)>expected_bytes:raise DownloadError('oversized_input')
                    if int(length)<expected_bytes:raise _Transient('truncated_content_length')
                while row['bytes_received']<expected_bytes:
                    remaining()
                    block=response.read(min(1048576,expected_bytes-row['bytes_received']))
                    if not block:raise _Transient('truncated_input')
                    if len(block)>expected_bytes-row['bytes_received']:raise DownloadError('oversized_input')
                    out.write(block);h.update(block);row['bytes_received']+=len(block)
                remaining()
                if response.read(1):raise DownloadError('oversized_input')
                if h.hexdigest()!=expected_sha256:raise DownloadError('input_hash_mismatch')
                out.flush();os.fsync(out.fileno())
            remaining()
            # Atomic no-clobber promotion after COMPLETE size and hash verification.
            os.link(part,destination)
            row['outcome']='verified';record();return destination
        except urllib.error.HTTPError as exc:
            code='http_'+str(exc.code)
            retry=exc.code in (429,500,502,503,504)
        except urllib.error.URLError as exc:
            retry=isinstance(exc.reason,(TimeoutError,ConnectionError,socket.timeout))
            code='network_interrupted' if retry else 'network_nontransient'
        except (_Transient,TimeoutError,ConnectionError,http.client.IncompleteRead,http.client.RemoteDisconnected) as exc:
            retry=True;code=str(exc) if isinstance(exc,_Transient) else 'transfer_interrupted'
        except DownloadError as exc:
            retry=False;code=str(exc)
        except Exception:
            row['outcome']='local_nontransient_error';record()
            raise DownloadError('local_download_error') from None
        row['outcome']=code;record()
        if not retry:raise DownloadError(code) from None
        if attempt==3:raise DownloadError('transfer_attempts_exhausted') from None
        pause=min(0.5*attempt,remaining());time.sleep(pause)
    raise DownloadError('unreachable_download_state')