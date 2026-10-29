"""Bounded four-stream private GCS ranges; exact SHA256; no redirects or retries.

Every range is requested once. A failed partial file is retained for diagnosis,
never accepted or resumed. No account credentials or capability URLs are logged.
"""
import concurrent.futures as futures
import http.client
import math
from pathlib import Path
import queue
import re
import socket
import threading
import time
from scoped_transport import validate_capability,deadline,digest

RANGE_BYTES=2*1024**2
STREAMS=4
IO_TIMEOUT=10.0

def download(cap,path,*,size,sha256,absolute_deadline,progress=None,on_progress=None,cap_validator=validate_capability):
    u,expiry=cap_validator(cap,'GET')
    path=Path(path)
    if (type(size) is not int or not 0<size<512*1024**2 or not isinstance(sha256,str)
        or re.fullmatch('[0-9a-f]{64}',sha256) is None or not math.isfinite(absolute_deadline)
        or path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents)):
        raise ValueError('Input bounds or unsafe destination')
    begun=time.monotonic();end=min(absolute_deadline,begun+expiry-time.time())
    if end<=begun:raise TimeoutError('No transfer budget')
    ranges=queue.Queue()
    for start in range(0,size,RANGE_BYTES):ranges.put((start,min(size-1,start+RANGE_BYTES-1)))
    workers=min(STREAMS,ranges.qsize())
    cancel=threading.Event();lock=threading.Lock();connections=[]
    counters={'bytes_received':0,'range_count':0,'connection_count':0,'request_count':0}
    state=progress if progress is not None else {}
    def report(phase):
        with lock:snapshot=dict(counters)
        state.update(snapshot,phase=phase,expected_bytes=size,parallel_streams=workers,
                     elapsed_seconds=round(time.monotonic()-begun,3),automatic_retries=0,
                     hash_verified=False,partial_file_valid=False)
        if on_progress:on_progress(dict(state))
    def remaining():
        value=end-time.monotonic()
        if value<=0:raise TimeoutError('Transfer absolute deadline reached')
        if cancel.is_set():raise InterruptedError('Transfer cancelled after peer failure')
        return min(IO_TIMEOUT,value)
    def abort():
        cancel.set()
        with lock:active=list(connections)
        for conn in active:
            try:
                if conn.sock is not None:conn.sock.shutdown(socket.SHUT_RDWR)
            except (OSError,AttributeError):pass
            try:conn.close()
            except Exception:pass
    # One fresh file; positional writes under a lock avoid overlapping writes.
    with path.open('xb') as out:
        out.truncate(size)
        def stream():
            conn=http.client.HTTPSConnection(u.hostname,timeout=remaining())
            with lock:connections.append(conn)
            try:
                while not cancel.is_set():
                    try:start,last=ranges.get_nowait()
                    except queue.Empty:return
                    timeout=remaining();conn.timeout=timeout
                    if conn.sock is not None:conn.sock.settimeout(timeout)
                    with lock:
                        counters['request_count']+=1
                        if conn.sock is None:counters['connection_count']+=1
                    headers={**cap['headers'],'Range':f'bytes={start}-{last}','Accept-Encoding':'identity'}
                    conn.request('GET',u.path+'?'+u.query,headers=headers)
                    response=conn.getresponse();n=last-start+1
                    if (response.status!=206 or response.getheader('Content-Range')!=f'bytes {start}-{last}/{size}'
                        or response.getheader('Content-Length')!=str(n)
                        or response.getheader('Content-Encoding') not in (None,'identity')):
                        raise ValueError('Range response binding failed')
                    received=0
                    while received<n:
                        timeout=remaining()
                        if conn.sock is not None:conn.sock.settimeout(timeout)
                        chunk=response.read(min(65536,n-received))
                        if not chunk:raise ValueError('Incomplete range')
                        with lock:
                            out.seek(start+received);out.write(chunk)
                            counters['bytes_received']+=len(chunk)
                        received+=len(chunk)
                    if response.read(1):raise ValueError('Range overflow')
                    remaining()
                    with lock:counters['range_count']+=1
            finally:conn.close()
        pool=futures.ThreadPoolExecutor(max_workers=workers,thread_name_prefix='private-range')
        jobs=[]
        try:
            report('started');jobs=[pool.submit(stream) for _ in range(workers)]
            pending=set(jobs);last_report=time.monotonic()
            while pending:
                budget=remaining()
                done,pending=futures.wait(pending,timeout=min(.1,budget),return_when=futures.FIRST_COMPLETED)
                for job in done:job.result()
                if time.monotonic()-last_report>=5:report('receiving');last_report=time.monotonic()
            pool.shutdown(wait=True)
            out.flush()
            remaining()
            with deadline(end):
                if counters['bytes_received']!=size or digest(path)!=sha256:raise ValueError('Full input hash failed')
        except BaseException as error:
            abort()
            for job in jobs:job.cancel()
            pool.shutdown(wait=True,cancel_futures=True)
            state['error_type']=type(error).__name__
            report('failed')
            raise
    state.update(phase='complete',bytes=size,bytes_received=size,expected_bytes=size,
                 sha256=sha256,hash_verified=True,partial_file_valid=True,
                 elapsed_seconds=round(time.monotonic()-begun,3),automatic_retries=0,
                 parallel_streams=workers,range_count=counters['range_count'],
                 connection_count=counters['connection_count'],request_count=counters['request_count'])
    if on_progress:on_progress(dict(state))
    return dict(state)