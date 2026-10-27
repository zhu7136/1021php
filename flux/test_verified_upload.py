"""CPU fault injection for119: no network, GPU or task APIs."""
import hashlib,http.client,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from flux import verified_upload as u
URL='https://storage.googleapis.com/fixture/object?signature=DO_NOT_LOG'
DATA=b'complete-payload'*1000
class Connection:
    def __init__(self,status=200,error=None,mutate=None,clock=None):
        self.status=status;self.error=error;self.sock=None;self.sent=b'';self.closed=False;self.mutate=mutate;self.clock=clock
    def request(self,method,path,body,headers):
        self.method=method;self.path=path;self.headers=headers
        while True:
            b=body.read(1024)
            if not b:break
            self.sent+=b
        if self.clock:self.clock[0]=181
    def getresponse(self):
        if self.mutate:self.mutate()
        if self.error:raise self.error
        return self
    def read(self,n):return b''
    def close(self):self.closed=True
class VerifiedUpload(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(dir='/workspace',prefix='put-test-'));self.file=self.root/'payload';self.file.write_bytes(DATA);self.events=[]
    def call(self,connections,url=URL,headers=None):
        with patch.object(u.http.client,'HTTPSConnection',side_effect=connections) as op,patch.object(u.time,'sleep'):
            r=u.upload(url,headers or {},self.file,on_event=lambda event,**kw:self.events.append({'event':event,**kw}))
        return r,op.call_count
    def fail(self,connections,code,**kwargs):
        with patch.object(u.http.client,'HTTPSConnection',side_effect=connections) as op,patch.object(u.time,'sleep'):
            with self.assertRaisesRegex(u.UploadError,code):u.upload(kwargs.get('url',URL),kwargs.get('headers',{}),self.file,on_event=lambda event,**kw:self.events.append({'event':event,**kw}))
        self.assertNotIn('DO_NOT_LOG',json.dumps(self.events));return op.call_count
    def test_success(self):
        c=Connection();r,n=self.call([c]);self.assertEqual(n,1);self.assertEqual(c.sent,DATA);self.assertEqual(c.method,'PUT');self.assertTrue(c.closed);self.assertEqual(r['sha256'],hashlib.sha256(DATA).hexdigest())
    def test_exact119_remote_disconnect_reopens_at_zero(self):
        a=Connection(error=http.client.RemoteDisconnected('DO_NOT_LOG'));b=Connection();r,n=self.call([a,b]);self.assertEqual(n,2);self.assertEqual(a.sent,DATA);self.assertEqual(b.sent,DATA);self.assertEqual(a.path,b.path);self.assertEqual(r['task_retries'],0);self.assertTrue(a.closed and b.closed);self.assertNotIn('DO_NOT_LOG',json.dumps(self.events))
    def test_connection_reset(self):self.assertEqual(self.call([Connection(error=ConnectionResetError()),Connection()])[1],2)
    def test_timeout(self):self.assertEqual(self.call([Connection(error=TimeoutError()),Connection()])[1],2)
    def test_incomplete_read(self):self.assertEqual(self.call([Connection(error=http.client.IncompleteRead(b'',1)),Connection()])[1],2)
    def test_503_then_success(self):self.assertEqual(self.call([Connection(status=503),Connection()])[1],2)
    def test_429_then_success(self):self.assertEqual(self.call([Connection(status=429),Connection()])[1],2)
    def test_500_502_504(self):
        for code in (500,502,504):self.assertEqual(self.call([Connection(status=code),Connection()])[1],2)
    def test_attempt_limit(self):self.assertEqual(self.fail([Connection(error=http.client.RemoteDisconnected()) for _ in range(3)],'upload_attempts_exhausted'),3)
    def test_auth_and_notfound_no_retry(self):
        for code in (400,401,403,404):self.assertEqual(self.fail([Connection(status=code)],'http_'+str(code)),1)
    def test_redirect_never_followed(self):self.assertEqual(self.fail([Connection(status=302)],'http_302'),1)
    def test_local_error_no_retry(self):self.assertEqual(self.fail([Connection(error=ValueError('DO_NOT_LOG'))],'nontransient_upload_error'),1)
    def test_untrusted_url(self):
        for url in ['http://storage.googleapis.com/x','https://evil.invalid/x','https://user:pass@storage.googleapis.com/x','https://storage.googleapis.com:444/x']:self.assertEqual(self.fail([],'untrusted_upload_host',url=url),0)
    def test_account_headers_rejected(self):
        for h in ['Authorization','Cookie','Host','Transfer-Encoding','Content-Length']:self.assertEqual(self.fail([],'invalid_upload_headers',headers={h:'DO_NOT_LOG'}),0)
    def test_source_mutation_blocks_retry(self):
        c=Connection(error=http.client.RemoteDisconnected(),mutate=lambda:self.file.write_bytes(b'x'*len(DATA)))
        self.assertEqual(self.fail([c],'upload_source_changed'),1)
    def test_shared_deadline(self):
        clock=[0]
        with patch.object(u.time,'monotonic',side_effect=lambda:clock[0]):self.assertEqual(self.fail([Connection(clock=clock)],'upload_deadline_exceeded'),1)
    def test_actual_generated_worker_uses_bounded_upload(self):
        import os
        with patch.dict(os.environ,{'PHP_G1_CLIMB_KIND':'low'}):
            from flux.g1_climb_resume_contract import worker_source
            ns={'__name__':'fixture','__file__':str(Path(__file__).with_name('run_smoke.py'))};exec(worker_source(Path(__file__).parent),ns)
            with patch.object(u,'upload',return_value={'task_retries':0}) as op:
                ns['upload'](URL,{},self.file);op.assert_called_once_with(URL,{},self.file,on_event=ns['emit'])
if __name__=='__main__':unittest.main()