"""Renew only two already-bound run objects, no account credential in worker."""
import http.client,json,time
from scoped_transport import validate_capability

def read_pointer(cap):
    u,_=validate_capability(cap,'GET')
    conn=http.client.HTTPSConnection(u.hostname,timeout=15)
    try:
        conn.request('GET',u.path+'?'+u.query,headers=cap['headers'])
        r=conn.getresponse()
        if r.status!=200:raise ValueError('Pointer read failed')
        data=r.read(32769)
        if len(data)>32768:raise ValueError('Pointer too large')
        return json.loads(data)
    finally:conn.close()

def poll(caps,end):
    output_path=caps['output']['object_path'];relay_path=caps['relay']['object_path']
    while time.monotonic()<end:
        try:
            new=read_pointer(caps['relay'])
            if set(new)!={'output','relay'}:raise ValueError('Unexpected pointer fields')
            for name,method,path in [('output','PUT',output_path),('relay','GET',relay_path)]:
                _,expiry=validate_capability(new[name],method)
                if new[name]['object_path']!=path or expiry<time.time()+120:raise ValueError('Pointer binding or expiry failed')
            caps.update(new)
        except Exception:
            # Do not expose capability data; keep last valid caps, never change object.
            pass
        time.sleep(min(60,max(0,end-time.monotonic())))