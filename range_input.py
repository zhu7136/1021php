"""Hash-pinned non-overlapping ranges, reused TLS connection, durable progress; no retries."""
import http.client,time,re
from pathlib import Path
from scoped_transport import validate_capability,deadline,digest
def download(cap,path,*,size,sha256,absolute_deadline,progress=None,on_progress=None):
    u,expiry=validate_capability(cap,'GET');path=Path(path)
    if path.exists() or not 0<size<512*1024**2 or not re.fullmatch('[0-9a-f]{64}',sha256):raise ValueError('Input bounds')
    end=min(absolute_deadline,time.monotonic()+expiry-time.time());begun=time.monotonic();count=0;parts=0;connections=0;last=0
    state=progress if progress is not None else {}
    def report(phase,force=False):
        nonlocal last
        state.update(phase=phase,bytes_received=count,expected_bytes=size,range_count=parts,connection_count=connections,elapsed_seconds=round(time.monotonic()-begun,3),automatic_retries=0,hash_verified=False)
        now=time.monotonic()
        if on_progress and (force or now-last>=5):on_progress(dict(state));last=now
    conn=http.client.HTTPSConnection(u.hostname,timeout=min(25,max(.001,end-time.monotonic())))
    report('started',True)
    try:
        with deadline(end),path.open('xb') as out:
            for start in range(0,size,8*1024**2):
                last_byte=min(size-1,start+8*1024**2-1)
                if conn.sock is None:connections+=1
                conn.timeout=min(25,max(.001,end-time.monotonic()))
                if conn.sock is not None:conn.sock.settimeout(conn.timeout)
                headers={**cap['headers'],'Range':f'bytes={start}-{last_byte}'}
                conn.request('GET',u.path+'?'+u.query,headers=headers);r=conn.getresponse()
                state.update(http_status=r.status,range_start=start,range_end=last_byte)
                if r.status!=206 or r.getheader('Content-Range')!=f'bytes {start}-{last_byte}/{size}' or int(r.getheader('Content-Length','-1'))!=last_byte-start+1:raise ValueError('Range response binding failed')
                received=0;report('receiving',True)
                while received<last_byte-start+1:
                    b=r.read(min(65536,last_byte-start+1-received))
                    if not b:raise ValueError('Incomplete range')
                    out.write(b);received+=len(b);count+=len(b);report('receiving')
                if r.read(1):raise ValueError('Range overflow')
                parts+=1;out.flush();report('range_complete',True)
            if count!=size or digest(path)!=sha256:raise ValueError('Full input hash failed')
    except Exception as e:
        state['error_type']=type(e).__name__;report('failed',True);raise
    finally:conn.close()
    state.update(phase='complete',bytes=count,bytes_received=count,sha256=sha256,range_count=parts,connection_count=connections,elapsed_seconds=round(time.monotonic()-begun,3),automatic_retries=0,hash_verified=True)
    if on_progress:on_progress(dict(state))
    return dict(state)