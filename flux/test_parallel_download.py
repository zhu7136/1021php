"""CPU-only Range transport fault injection; no GPU or task submission."""
import hashlib,io,json,tempfile,threading,time,urllib.error,unittest
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
  self.fail(lambda *a,**kw:(_ for _ in ()).throw(TimeoutError('DO_NOT_LOG')),'transfer_or_local_error')
 def test_invalid_contract_not_contacted(self):
  for size in [True,0,101*1024**2]:
   with patch.object(d,'open_range') as op,self.assertRaises(d.DownloadError):d.download(URL,self.target,size,SHA,self.diag)
   op.assert_not_called()
if __name__=='__main__':unittest.main()