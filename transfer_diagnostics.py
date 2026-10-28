"""Bounded, single-request input download with credential-free diagnostics.

Uses the existing pinned-capability validation; never prints URLs or headers.
No retries, redirects, range requests or changes to payload integrity rules.
"""
import http.client
import json
import re
import time
from pathlib import Path
from scoped_transport import validate_capability, deadline, digest, MAX_BYTES


def download(cap, path, *, size, sha256, absolute_deadline, progress):
    started = time.monotonic()
    count = 0
    phase = 'validation'
    status = None
    def emit(event):
        elapsed = max(0, time.monotonic() - started)
        progress.update(event=event, phase=phase, bytes_received=count,
                        expected_bytes=size, elapsed_seconds=round(elapsed, 3),
                        mean_bytes_per_second=round(count / max(elapsed, .001), 1),
                        http_status=status, automatic_retries=0)
        print(json.dumps({**progress, 'transfer_event': event, 'event': 'input_transfer_progress'}), flush=True)
    conn = None
    try:
        with deadline(absolute_deadline):
            u, expiry = validate_capability(cap, 'GET')
            if type(size) is not int or not 0 < size <= MAX_BYTES:
                raise ValueError('Invalid size bound')
            if not re.fullmatch('[0-9a-f]{64}', sha256):
                raise ValueError('Invalid hash bound')
            path = Path(path)
            if path.exists() or path.is_symlink():
                raise FileExistsError('Fresh download required')
            end = min(absolute_deadline, time.monotonic() + expiry - time.time())
            with deadline(end):
                phase = 'request'
                emit('request_started')
                conn = http.client.HTTPSConnection(u.hostname, timeout=max(.001, min(30, end-time.monotonic())))
                conn.request('GET', u.path + '?' + u.query, headers=dict(cap['headers']))
                phase = 'response_headers'
                emit('request_sent')
                response = conn.getresponse()
                status = response.status
                if not 200 <= status < 300:
                    raise RuntimeError('Object transfer HTTP failure')
                declared = response.getheader('Content-Length')
                if declared is not None and int(declared) != size:
                    raise ValueError('Response length mismatch')
                phase = 'body'
                emit('headers_received')
                last = time.monotonic()
                with path.open('xb') as f:
                    while count < size:
                        b = response.read(min(65536, size-count))
                        if not b:
                            raise ValueError('Incomplete download')
                        f.write(b)
                        count += len(b)
                        if count == len(b) or time.monotonic()-last >= 5:
                            emit('body_progress')
                            last = time.monotonic()
                    if response.read(1):
                        raise ValueError('Oversized download')
                phase = 'sha256'
                emit('body_complete')
                if digest(path) != sha256:
                    raise ValueError('Downloaded SHA256 mismatch')
                phase = 'complete'
                emit('verified')
                return dict(progress, verified=True)
    except Exception as e:
        progress['error_type'] = type(e).__name__
        emit('failed')
        raise
    finally:
        if conn is not None:
            conn.close()