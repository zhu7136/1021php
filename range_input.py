"""Bounded non-overlapping ranges of one private signed object. No retries."""
import http.client,time,re
from pathlib import Path
from scoped_transport import validate_capability,deadline,digest
def download(cap,path,*,size,sha256,absolute_deadline):
 u,expiry=validate_capability(cap,'GET');path=Path(path)
 if path.exists() or not 0<size<512*1024**2 or not re.fullmatch('[0-9a-f]{64}',sha256):raise ValueError('Input bounds')
 end=min(absolute_deadline,time.monotonic()+expiry-time.time());count=0;begun=time.monotonic();parts=0
 with deadline(end),path.open('xb') as out:
  for start in range(0,size,8*1024**2):
   last=min(size-1,start+8*1024**2-1);headers={**cap['headers'],'Range':f'bytes={start}-{last}'}
   conn=http.client.HTTPSConnection(u.hostname,timeout=min(25,max(.001,end-time.monotonic())))
   try:
    conn.request('GET',u.path+'?'+u.query,headers=headers);r=conn.getresponse()
    if r.status!=206 or r.getheader('Content-Range')!=f'bytes {start}-{last}/{size}' or int(r.getheader('Content-Length','-1'))!=last-start+1:raise ValueError('Range response binding failed')
    received=0
    while received<last-start+1:
     b=r.read(min(65536,last-start+1-received))
     if not b:raise ValueError('Incomplete range')
     out.write(b);received+=len(b);count+=len(b)
    if r.read(1):raise ValueError('Range overflow')
   finally:conn.close()
   parts+=1
 if count!=size or digest(path)!=sha256:raise ValueError('Full input hash failed')
 return {'bytes':count,'sha256':sha256,'range_count':parts,'elapsed_seconds':round(time.monotonic()-begun,3),'automatic_retries':0,'hash_verified':True}