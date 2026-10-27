"""File-byte progress only: never emit URLs, headers or capability fields."""
import json
from pathlib import Path
import threading
import time


def monitored_download(operation,cap,path,*,state,emit,interval=5,**kwargs):
    if kwargs.get('method')!='GET' or interval<=0:raise ValueError('GET progress required')
    started=time.monotonic();stop=threading.Event();path=Path(path)
    expected=kwargs['size'];lock=threading.Lock()
    def snapshot(status):
        try:count=path.stat().st_size if path.is_file() else 0
        except OSError:count=0
        elapsed=time.monotonic()-started
        data=dict(event='hud04_download_progress',status=status,bytes_received_file=count,
            expected_bytes=expected,elapsed_seconds=round(elapsed,3),
            file_bytes_per_second=round(count/elapsed,2) if elapsed>0 else 0,
            sha256_verified=status=='verified',automatic_retries=0,
            byte_measurement='local_file_size_not_network_wire_bytes')
        with lock:state.clear();state.update(data)
        emit(data)
    def watch():
        while not stop.wait(interval):snapshot('in_progress')
    snapshot('starting')
    thread=threading.Thread(target=watch,name='download-file-progress',daemon=True);thread.start()
    try:
        receipt=operation(cap,path,**kwargs)
        stop.set();thread.join(timeout=1)
        if receipt.get('verified') is not True:raise ValueError('Missing transfer verification')
        snapshot('verified');return receipt
    except BaseException as error:
        stop.set();thread.join(timeout=1);snapshot('failed')
        state['error_type']=type(error).__name__
        raise
    finally:
        stop.set();thread.join(timeout=1)