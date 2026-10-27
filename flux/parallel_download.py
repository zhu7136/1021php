"""Four bounded Range GETs, no retries, hash-verified atomic publication.

Only scoped GCS URLs are accepted. Every response must prove its exact byte
range and object ETag. Errors, partials and timeouts never publish an input.
"""
import hashlib,json,os,re,threading,time,urllib.error,urllib.request,uuid
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from flux.verified_download import check_url,_ScopedRedirect,DownloadError

WORKERS=4
CHUNK_BYTES=2*1024**2
DEADLINE_SECONDS=180
SOCKET_SECONDS=5

def open_range(url,start,end,timeout,etag=None):
 headers={'Accept-Encoding':'identity','Range':f'bytes={start}-{end}'}
 if etag is not None:headers['If-Match']=etag
 request=urllib.request.Request(url,headers=headers)
 return urllib.request.build_opener(_ScopedRedirect()).open(request,timeout=timeout)

def download(url,destination,expected_bytes,expected_sha256,diagnostics):
 check_url(url)
 if (type(expected_bytes) is not int or not 1<=expected_bytes<=100*1024**2 or
     not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256)):
  raise DownloadError('invalid_download_contract')
 destination=Path(destination);diagnostics=Path(diagnostics)
 if destination.exists() or diagnostics.exists():raise DownloadError('download_path_already_exists')
 started=time.monotonic();deadline=started+DEADLINE_SECONDS
 stop=threading.Event();lock=threading.Lock();rows=[];published=False;failure=None
 token=uuid.uuid4().hex[:10]
 def remaining():
  if stop.is_set():raise DownloadError('download_cancelled')
  left=deadline-time.monotonic()
  if left<=0:raise DownloadError('download_deadline_exceeded')
  return left
 def checked(response,start,end,etag=None):
  check_url(response.url)
  if response.status!=206:raise DownloadError('range_response_required')
  if response.headers.get('Content-Range')!=f'bytes {start}-{end}/{expected_bytes}':
   raise DownloadError('wrong_content_range')
  if response.headers.get('Content-Length')!=str(end-start+1):raise DownloadError('wrong_content_length')
  if response.headers.get('Content-Encoding','identity').lower()!='identity':raise DownloadError('encoded_range_rejected')
  actual=response.headers.get('ETag')
  if not actual or actual.startswith('W/') or '\n' in actual or '\r' in actual:raise DownloadError('strong_etag_required')
  if etag is not None and actual!=etag:raise DownloadError('object_changed')
  return actual
 def read(response,n):
  remaining()
  # read1 returns after one underlying read, unlike read(n) which can spend the
  # whole shared budget filling one large block on a slow connection.
  block=response.read1(n)
  remaining()
  return block
 def transfer(index,start,end,etag):
  row={'index':index,'start':start,'end':end,'bytes_received':0,'outcome':'started'}
  with lock:rows.append(row)
  part=destination.with_name(destination.name+f'.range-{token}-{index}.part')
  try:
   with open_range(url,start,end,min(SOCKET_SECONDS,remaining()),etag) as response:
    checked(response,start,end,etag)
    with part.open('xb') as target:
     while row['bytes_received']<end-start+1:
      block=read(response,min(65536,end-start+1-row['bytes_received']))
      if not block:raise DownloadError('truncated_range')
      if len(block)>end-start+1-row['bytes_received']:raise DownloadError('oversized_range')
      target.write(block);row['bytes_received']+=len(block)
     if read(response,1):raise DownloadError('oversized_range')
   row['outcome']='received';return index,part
  except BaseException:
   row['outcome']='failed';stop.set();raise
 try:
  # Probe byte-range support and bind all parallel reads to one object version.
  with open_range(url,0,0,min(SOCKET_SECONDS,remaining())) as response:
   etag=checked(response,0,0)
   if len(read(response,1))!=1 or read(response,1):raise DownloadError('invalid_range_probe')
  ranges=[(i,start,min(expected_bytes-1,start+CHUNK_BYTES-1)) for i,start in enumerate(range(0,expected_bytes,CHUNK_BYTES))]
  parts={}
  with ThreadPoolExecutor(max_workers=WORKERS) as executor:
   futures=[executor.submit(transfer,i,start,end,etag) for i,start,end in ranges]
   try:
    for future in as_completed(futures):
     index,part=future.result();parts[index]=part
   except BaseException:
    stop.set()
    for future in futures:future.cancel()
    raise
  remaining();assembled=destination.with_name(destination.name+'.assembled-'+token+'.part')
  h=hashlib.sha256();count=0
  with assembled.open('xb') as target:
   for index,_,_ in ranges:
    with parts[index].open('rb') as source:
     while True:
      remaining();block=source.read(1048576)
      if not block:break
      target.write(block);h.update(block);count+=len(block)
   target.flush();os.fsync(target.fileno())
  if count!=expected_bytes:raise DownloadError('assembled_size_mismatch')
  if h.hexdigest()!=expected_sha256:raise DownloadError('input_hash_mismatch')
  remaining();os.link(assembled,destination);published=True
  return destination
 except urllib.error.HTTPError as exc:
  failure='http_'+str(exc.code);raise DownloadError(failure) from None
 except DownloadError as exc:
  failure=str(exc);raise
 except BaseException:
  failure='transfer_or_local_error';raise DownloadError(failure) from None
 finally:
  stop.set()
  diagnostics.write_text(json.dumps({'mode':'parallel_range','workers':WORKERS,'chunk_bytes':CHUNK_BYTES,
   'deadline_seconds':DEADLINE_SECONDS,'socket_timeout_seconds':SOCKET_SECONDS,
   'elapsed_seconds':round(time.monotonic()-started,3),'expected_bytes':expected_bytes,
   'expected_sha256':expected_sha256,'task_retries':0,'transfer_retries':0,
   'published_verified':published,'failure':failure,'chunks':sorted(rows,key=lambda x:x['index'])},indent=2))