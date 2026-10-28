"""CPU-only Range transport fault injection; no GPU or task submission."""
import errno,hashlib,http.client,io,json,socket,ssl,tempfile,threading,time,urllib.error,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from flux import parallel_download as d
URL='https://storage.googleapis.com/fixture/object?signature=DO_NOT_LOG'
DATA=b'abcdefghijklmnopqrstuvwxyz0123456789';SHA=hashlib.sha256(DATA).hexdigest()
class Response:
 def __init__(self,start,end,*,data=DATA,etag='"v1"',status=206):
  self.url=URL;self.status=status;self.body=io.BytesIO(data[start:end+1]);self.headers={'Content-Range':f'bytes {start}-{end}/{len(DATA)}','Content-Length':str(end-start+1),'ETag':etag}
 def __enter__(self):return self
 def __exit__(self,*args):return False
 def read1(self,n):return self.body.read(n)
class ParallelDownloadTests(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(dir='/workspace',prefix='parallel-transfer-test-'));self.target=self.root/'input.tar.gz';self.diag=self.root/'report.json'
 def invoke(self,opener,expected=SHA):
  with patch.object(d,'open_range',side_effect=opener),patch.object(d,'CHUNK_BYTES',8):
   return d.download(URL,self.target,len(DATA),expected,self.diag)
 def success(self,opener):
  self.invoke(opener);self.assertEqual(self.target.read_bytes(),DATA)
  report=json.loads(self.diag.read_text());self.assertTrue(report['published_verified']);self.assertEqual(report['transfer_retries'],0)
  self.assertNotIn('DO_NOT_LOG',self.diag.read_text());return report
 def fail(self,opener,reason=None):
  with self.assertRaises(d.DownloadError) as cm:self.invoke(opener)
  if reason:self.assertIn(reason,str(cm.exception))
  self.assertFalse(self.target.exists());self.assertNotIn('DO_NOT_LOG',self.diag.read_text())
 def test_success_and_exact_ranges(self):
  calls=[]
  def op(url,start,end,timeout,etag=None):
   calls.append((start,end,etag));self.assertLessEqual(timeout,5);return Response(start,end)
  report=self.success(op);self.assertEqual(len(calls),6);self.assertEqual(sum(r['bytes_received'] for r in report['chunks']),len(DATA))
  self.assertTrue(all(e=='"v1"' for s,_,e in calls if (s,_)!=(0,0)))
 def test_four_concurrent_workers(self):
  lock=threading.Lock();active=[0];peak=[0]
  def op(url,start,end,timeout,etag=None):
   if etag:
    with lock:active[0]+=1;peak[0]=max(peak[0],active[0])
    time.sleep(.02)
    with lock:active[0]-=1
   return Response(start,end)
  self.success(op);self.assertEqual(peak[0],4)
 def test_server_ignores_range(self):self.fail(lambda u,s,e,t,etag=None:Response(s,e,status=200),'range_response_required')
 def test_wrong_total(self):
  def op(u,s,e,t,etag=None):
   r=Response(s,e);r.headers['Content-Range']=f'bytes {s}-{e}/999';return r
  self.fail(op,'wrong_content_range')
 def test_wrong_range(self):
  def op(u,s,e,t,etag=None):
   r=Response(s,e);r.headers['Content-Range']=f'bytes {s+1}-{e+1}/{len(DATA)}';return r
  self.fail(op,'wrong_content_range')
 def test_wrong_length(self):
  def op(u,s,e,t,etag=None):
   r=Response(s,e);r.headers['Content-Length']='999';return r
  self.fail(op,'wrong_content_length')
 def test_encoded_response(self):
  def op(u,s,e,t,etag=None):
   r=Response(s,e);r.headers['Content-Encoding']='gzip';return r
  self.fail(op,'encoded_range_rejected')
 def test_missing_etag(self):self.fail(lambda u,s,e,t,etag=None:Response(s,e,etag=None),'strong_etag_required')
 def test_weak_etag(self):self.fail(lambda u,s,e,t,etag=None:Response(s,e,etag='W/"x"'),'strong_etag_required')
 def test_object_changed_midtransfer(self):self.fail(lambda u,s,e,t,etag=None:Response(s,e,etag='"v2"' if etag else '"v1"'))
 def test_short_chunk_not_published(self):
  def op(u,s,e,t,etag=None):
   r=Response(s,e)
   if etag:r.body=io.BytesIO(b'')
   return r
  self.fail(op)
 def test_oversize_probe(self):
  def op(u,s,e,t,etag=None):
   r=Response(s,e);r.body=io.BytesIO(b'xx');return r
  self.fail(op,'invalid_range_probe')
 def test_wrong_hash_never_published(self):
  with self.assertRaisesRegex(d.DownloadError,'input_hash_mismatch'):self.invoke(lambda u,s,e,t,etag=None:Response(s,e),expected='0'*64)
  self.assertFalse(self.target.exists())
 def test_auth_failure_no_retry(self):
  calls=[]
  def op(*a,**kw):calls.append(1);raise urllib.error.HTTPError(URL,403,'sensitive',{},None)
  self.fail(op,'http_403');self.assertEqual(len(calls),1)
 def test_deadline_while_reading(self):
  clock=[0]
  class Slow(Response):
   def read1(self,n):clock[0]=181;return super().read1(n)
  with patch.object(d.time,'monotonic',side_effect=lambda:clock[0]):self.fail(lambda u,s,e,t,etag=None:Slow(s,e),'download_deadline_exceeded')
 def test_existing_input_never_overwritten(self):
  self.target.write_bytes(b'keep')
  with patch.object(d,'open_range') as op,self.assertRaises(d.DownloadError):d.download(URL,self.target,len(DATA),SHA,self.diag)
  op.assert_not_called();self.assertEqual(self.target.read_bytes(),b'keep')
 def test_untrusted_host_not_contacted(self):
  with patch.object(d,'open_range') as op,self.assertRaises(d.DownloadError):d.download('https://evil.invalid/object',self.target,len(DATA),SHA,self.diag)
  op.assert_not_called()
 def test_secret_exception_not_in_diagnostics(self):
  self.fail(lambda *a,**kw:(_ for _ in ()).throw(TimeoutError('DO_NOT_LOG')),'socket_timeout')
 def test_safe_error_taxonomy(self):
  cases=[(TimeoutError(URL),'socket_timeout'),(urllib.error.URLError(TimeoutError(URL)),'socket_timeout'),
   (urllib.error.URLError(URL),'network_url_error'),(ssl.SSLError(URL),'tls_error'),
   (socket.gaierror(-2,URL),'dns_error'),(ConnectionResetError(URL),'connection_error'),
   (http.client.IncompleteRead(b'DO_NOT_LOG',99),'incomplete_read'),
   (OSError(errno.ENOSPC,URL),'local_no_space'),(PermissionError(errno.EACCES,URL),'local_permission_denied'),
   (OSError(errno.EMFILE,URL),'local_fd_limit'),(OSError(errno.EROFS,URL),'local_read_only'),
   (OSError(errno.EXDEV,URL),'local_cross_device'),(OSError(errno.EIO,URL),'local_io_error'),
   (ValueError(URL),'unexpected_error'),(d.DownloadError('secret_alpha_only'),'unexpected_error')]
  for exc,code in cases:
   with self.subTest(code=code):self.assertEqual(d.error_code(exc),code)
 def test_probe_timeout_is_localized(self):
  def fail(*args,**kw):raise urllib.error.URLError(TimeoutError(URL))
  self.fail(fail,'socket_timeout');r=json.loads(self.diag.read_text())
  self.assertEqual(r['failure_stage'],'range_probe');self.assertEqual(r['chunks'],[])
 def test_chunk_timeout_is_localized(self):
  class Timed(Response):
   def read1(self,n):raise TimeoutError(URL)
  self.fail(lambda u,s,e,t,etag=None:Timed(s,e) if etag else Response(s,e),'socket_timeout')
  r=json.loads(self.diag.read_text());self.assertEqual(r['failure_stage'],'range_read')
  self.assertTrue(any(c.get('failure')=='socket_timeout' for c in r['chunks']))
 def test_cancelled_future_cannot_hide_primary_timeout(self):
  def op(u,s,e,t,etag=None):
   if etag:raise TimeoutError(URL)
   return Response(s,e)
  with patch.object(d,'WORKERS',1),patch.object(d,'as_completed',side_effect=lambda fs:iter(fs[::-1])):
   self.fail(op,'socket_timeout')
  self.assertEqual(json.loads(self.diag.read_text())['failure'],'socket_timeout')
 def test_chunk_connection_reset(self):
  def op(u,s,e,t,etag=None):
   if etag:raise ConnectionResetError(URL)
   return Response(s,e)
  self.fail(op,'connection_error');self.assertEqual(json.loads(self.diag.read_text())['failure_stage'],'range_open')
 def test_part_open_disk_full(self):
  original=Path.open
  def op(path,*args,**kw):
   if '.range-' in path.name:raise OSError(errno.ENOSPC,URL)
   return original(path,*args,**kw)
  with patch.object(Path,'open',op):self.fail(lambda u,s,e,t,etag=None:Response(s,e),'local_no_space')
  self.assertEqual(json.loads(self.diag.read_text())['failure_stage'],'part_open')
 def test_publication_permission_error(self):
  with patch.object(d.os,'link',side_effect=PermissionError(errno.EACCES,URL)):
   self.fail(lambda u,s,e,t,etag=None:Response(s,e),'local_permission_denied')
  self.assertEqual(json.loads(self.diag.read_text())['failure_stage'],'publication')
 def test_wrapped_tls_failure_no_retry(self):
  calls=[]
  def op(*args,**kw):calls.append(1);raise urllib.error.URLError(ssl.SSLError(URL))
  self.fail(op,'tls_error');self.assertEqual(len(calls),1)
 def test_five_input_nested_parallelism(self):
  from flux.distill_diagnostics import monitored_download
  from flux.student_video_contract import input_jobs
  control={'url':URL,'bytes':len(DATA),'sha256':SHA};jobs=input_jobs([control]*4,control)
  lock=threading.Lock();active=0;peak=0;events=[]
  def op(u,s,e,t,etag=None):
   nonlocal active,peak
   if etag:
    with lock:active+=1;peak=max(peak,active)
    time.sleep(.01)
    with lock:active-=1
   return Response(s,e)
  def emit(event,**kw):
   with lock:events.append(dict(event=event,**kw))
  def get(job):return monitored_download(job[0],job[1],self.root,self.root/'evidence',d.download,emit)
  with patch.object(d,'open_range',side_effect=op),patch.object(d,'CHUNK_BYTES',8),ThreadPoolExecutor(max_workers=3) as pool:
   paths=list(pool.map(get,jobs))
  self.assertEqual(len(paths),5);self.assertTrue(all(p.read_bytes()==DATA for p in paths))
  self.assertLessEqual(peak,12);self.assertGreaterEqual(peak,4)
  self.assertEqual(sum(e['event']=='five_distill_input_verified' for e in events),5)
  self.assertNotIn('DO_NOT_LOG',json.dumps(events))
 def test_invalid_contract_not_contacted(self):
  for size in [True,0,101*1024**2]:
   with patch.object(d,'open_range') as op,self.assertRaises(d.DownloadError):d.download(URL,self.target,size,SHA,self.diag)
   op.assert_not_called()
if __name__=='__main__':unittest.main()