"""Single-GET, staged diagnostics for exactly the five r46-unverified wheels."""
import hashlib,http.client,json,os,time,traceback
from pathlib import Path
from urllib.parse import urlparse
from fetch_official_wheels import absolute_budget,load_manifest

PIN='6d0833e158ccb8a0fb3ac4776a467a1ac2a45ef72c99caa7e59da5ad38c96049'
NAMES={'wandb','xatlas','xxhash','yourdfpy','onnx'}
def selected(path):
 manifest=load_manifest(path,PIN);rows=[r for r in manifest['files'] if r['name'] in NAMES]
 if len(rows)!=5 or {r['name'] for r in rows}!=NAMES:raise ValueError('wrong incremental scope')
 return rows

def fetch_one(row,directory,deadline,emit,connection_factory=http.client.HTTPSConnection):
 start=time.monotonic();stage='connect';count=0;digest=hashlib.sha256();connection=None
 report=dict(filename=row['filename'],expected_bytes=row['bytes'],expected_sha256=row['sha256'],automatic_retries=0,passed=False)
 def event(kind,**extra):emit('hud04_incremental_'+kind,filename=row['filename'],stage=stage,received_bytes=count,elapsed_seconds=round(time.monotonic()-start,3),remaining_seconds=max(0,round(deadline-time.monotonic(),3)),**extra)
 def timeout():
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('file absolute deadline')
  return min(15,remaining)
 def update_socket():
  if connection.sock is not None:connection.sock.settimeout(timeout())
 try:
  with absolute_budget(max(0.001,deadline-time.monotonic())):
   event('file_start');url=urlparse(row['url']);connection=connection_factory(url.hostname,timeout=timeout());connection.connect();stage='request';event('connected')
   update_socket();connection.request('GET',url.path,headers={'Accept-Encoding':'identity','User-Agent':'hud04-staged-transfer/1'});stage='response_headers';update_socket();response=connection.getresponse();report['http_status']=response.status
   if response.status!=200:raise ValueError('HTTP status or redirect rejected')
   content_length=response.getheader('Content-Length');report['declared_bytes']=int(content_length) if content_length is not None else None
   if content_length is not None and int(content_length)!=row['bytes']:raise ValueError('declared size mismatch')
   if response.getheader('Content-Encoding') not in [None,'identity']:raise ValueError('unexpected encoding')
   event('headers',http_status=response.status,declared_bytes=report['declared_bytes']);stage='body_read';last_progress=start;last_count=0
   target=directory/(row['filename']+'.part')
   with target.open('xb') as output:
    while True:
     update_socket();chunk=response.read1(min(65536,row['bytes']-count+1))
     if not chunk:break
     count+=len(chunk)
     if count>row['bytes']:raise ValueError('excess body')
     digest.update(chunk);output.write(chunk)
     now=time.monotonic()
     if count-last_count>=1024*1024 or now-last_progress>=5:event('progress');last_progress=now;last_count=count
   stage='verify'
   if count!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('size or SHA256 mismatch')
   final=directory/row['filename']
   if final.exists() or final.is_symlink():raise ValueError('refuse replacement')
   target.rename(final);stage='complete';report.update(passed=True,sha256=digest.hexdigest())
 except Exception as error:
  report.update(error_type=type(error).__name__,failed_stage=stage,traceback_frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-8:]])
 finally:
  if connection is not None:connection.close()
 report.update(received_bytes=count,elapsed_seconds=round(time.monotonic()-start,3),stage=stage)
 event('file_result',result=report);return report

def run(rows,output,deadline,emit,connection_factory=http.client.HTTPSConnection):
 output=Path(output).absolute()
 if any(p.is_symlink() for p in [output,*output.parents]):raise ValueError('symlink path')
 output.mkdir(exist_ok=False);directory=output/'wheels';directory.mkdir();records=[]
 for row in rows:
  remaining=deadline-time.monotonic()
  if remaining<2:
   record=dict(filename=row['filename'],passed=False,attempted=False,error_type='InsufficientRemainingBudget',received_bytes=0,automatic_retries=0);emit('hud04_incremental_skipped',result=record)
  else:
   limit=60 if row['name'] in ['wandb','onnx'] else 20
   record=fetch_one(row,directory,min(deadline,time.monotonic()+limit),emit,connection_factory);record['attempted']=True
  records.append(record)
  (output/'partial-receipt.json').write_text(json.dumps(dict(records=records,complete=False,manifest_sha256=PIN),indent=2)+'\n')
 result=dict(records=records,complete=len(records)==5 and all(r['passed'] for r in records),files_requested=5,files_verified=sum(r['passed'] for r in records),verified_bytes=sum(r['received_bytes'] for r in records if r['passed']),automatic_retries=0,installed=False,downloaded_code_imported=False)
 (output/'download-receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result