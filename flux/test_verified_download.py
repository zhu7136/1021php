"""CPU-only injected transfer faults. No remote tasks or model calls."""
import hashlib,http.client,io,json,socket,tarfile,tempfile,urllib.error,unittest
from pathlib import Path
from unittest.mock import patch
from flux import verified_download as d

URL='https://storage.googleapis.com/fixture/object?signature=DO_NOT_LOG'
DATA=b'abcdefgh';SHA=hashlib.sha256(DATA).hexdigest()
class Response:
    def __init__(self,data=DATA,*,length=8,url=URL,error=None):
        self.body=io.BytesIO(data);self.headers={} if length is None else {'Content-Length':str(length)}
        self.url=url;self.status=200;self.error=error
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,n):
        if self.error:raise self.error
        return self.body.read(n)

class VerifiedDownload(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(dir='/workspace',prefix='download-test-'))
        self.target=self.root/'input.tar.gz';self.diag=self.root/'data-download.json'
    def run_download(self,responses,*,expected=DATA):
        with patch.object(d,'open_url',side_effect=responses) as open_mock,patch.object(d.time,'sleep'):
            path=d.download(URL,self.target,len(expected),hashlib.sha256(expected).hexdigest(),self.diag)
        self.assertEqual(path.read_bytes(),expected)
        self.assertNotIn('DO_NOT_LOG',self.diag.read_text())
        return json.loads(self.diag.read_text()),open_mock.call_count
    def test_success_one_attempt(self):
        report,calls=self.run_download([Response()]);self.assertEqual(calls,1)
        self.assertEqual(report['attempts'][0]['outcome'],'verified')
    def test_exact_296_premature_eof_then_success(self):
        report,calls=self.run_download([Response(b'abc'),Response()])
        self.assertEqual(calls,2);self.assertEqual(report['attempts'][0]['outcome'],'truncated_input')
        self.assertEqual(report['attempts'][0]['bytes_received'],3)
        self.assertEqual(report['attempts'][1]['bytes_received'],8)
    def test_connection_reset_then_success(self):
        _,calls=self.run_download([Response(error=ConnectionResetError('sensitive detail')),Response()]);self.assertEqual(calls,2)
    def test_timeout_then_success(self):
        _,calls=self.run_download([socket.timeout('sensitive'),Response()]);self.assertEqual(calls,2)
    def test_incomplete_read_then_success(self):
        _,calls=self.run_download([Response(error=http.client.IncompleteRead(b'ab',6)),Response()]);self.assertEqual(calls,2)
    def test_short_content_length_then_success(self):
        _,calls=self.run_download([Response(b'abc',length=3),Response()]);self.assertEqual(calls,2)
    def test_transient_http_then_success(self):
        _,calls=self.run_download([urllib.error.HTTPError(URL,503,'secret',{},None),Response()]);self.assertEqual(calls,2)
    def fail_download(self,responses,expected_code):
        with patch.object(d,'open_url',side_effect=responses) as op,patch.object(d.time,'sleep'):
            with self.assertRaisesRegex(d.DownloadError,expected_code):d.download(URL,self.target,8,SHA,self.diag)
        self.assertFalse(self.target.exists());self.assertNotIn('DO_NOT_LOG',self.diag.read_text())
        return op.call_count
    def test_three_truncations_stop_without_published_input(self):
        self.assertEqual(self.fail_download([Response(b'abc') for _ in range(3)],'transfer_attempts_exhausted'),3)
    def test_hash_mismatch_no_retry(self):self.assertEqual(self.fail_download([Response(b'12345678')],'input_hash_mismatch'),1)
    def test_oversize_body_no_retry(self):self.assertEqual(self.fail_download([Response(DATA+b'x')],'oversized_input'),1)
    def test_oversize_header_no_retry(self):self.assertEqual(self.fail_download([Response(length=9)],'oversized_input'),1)
    def test_http403_no_retry(self):self.assertEqual(self.fail_download([urllib.error.HTTPError(URL,403,'secret',{},None)],'http_403'),1)
    def test_http404_no_retry(self):self.assertEqual(self.fail_download([urllib.error.HTTPError(URL,404,'secret',{},None)],'http_404'),1)
    def test_untrusted_response_no_retry(self):self.assertEqual(self.fail_download([Response(url='https://evil.invalid/object')],'untrusted_download_host'),1)
    def test_untrusted_request_never_sent(self):
        for bad in ['http://storage.googleapis.com/x','https://user:secret@storage.googleapis.com/x','https://evil.invalid/x']:
            with patch.object(d,'open_url') as op,self.assertRaises(d.DownloadError):d.download(bad,self.target,8,SHA,self.diag)
            op.assert_not_called()
    def test_redirect_rejected_before_following(self):
        with self.assertRaises(d.DownloadError):d._ScopedRedirect().redirect_request(None,None,302,'',{},'https://evil.invalid/object')
    def test_deadline_no_unbounded_wait(self):
        clock=[0]
        class Slow(Response):
            def read(self,n):clock[0]=181;return super().read(n)
        with patch.object(d.time,'monotonic',side_effect=lambda:clock[0]):
            self.assertEqual(self.fail_download([Slow()],'download_deadline_exceeded'),1)
    def test_existing_input_not_overwritten(self):
        self.target.write_bytes(b'keep')
        with patch.object(d,'open_url') as op,self.assertRaisesRegex(d.DownloadError,'already_exists'):d.download(URL,self.target,8,SHA,self.diag)
        op.assert_not_called();self.assertEqual(self.target.read_bytes(),b'keep')
    def test_highstep_extraction_after_recovered_transfer(self):
        from flux import g1_highstep_input as inp
        raw=b'official fixture';member='motions-high-step/sample.txt';b=io.BytesIO()
        with tarfile.open(fileobj=b,mode='w:gz') as tar:
            info=tarfile.TarInfo(member);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        archive=b.getvalue();manifest={'components':{'motions-high-step':{'files':[{'path':member,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}}}
        (self.root/'release-assets.json').write_text(json.dumps(manifest))
        control={'url':URL,'bytes':len(archive),'sha256':hashlib.sha256(archive).hexdigest()}
        with patch.object(inp,'ROOT',self.root),patch.object(d,'open_url',side_effect=[Response(archive[:10],length=len(archive)),Response(archive,length=len(archive))]),patch.object(d.time,'sleep'):
            out=inp.fetch_data(control,self.root/'highstep-data')
        self.assertEqual((out/member).read_bytes(),raw)
    def test_checkpoint_uses_same_verified_transfer(self):
        from flux import g1_resume292_input as inp
        control={'url':URL,'bytes':8,'sha256':SHA,'source_task':inp.SOURCE_TASK,'source_run':inp.SOURCE_RUN}
        with patch.object(inp,'CHECKPOINT_BYTES',8),patch.object(inp,'CHECKPOINT_SHA256',SHA),patch.object(d,'open_url',side_effect=[Response(b'ab'),Response()]),patch.object(d.time,'sleep'):
            path=inp.fetch_checkpoint(control,self.root/'resume-input')
        self.assertEqual(path.read_bytes(),DATA)
        self.assertTrue((self.root/'checkpoint-download.json').is_file())

if __name__=='__main__':unittest.main()