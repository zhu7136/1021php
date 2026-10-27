"""CPU fault injection for sanitized, retained transport diagnostics."""
import json,tempfile,unittest
from pathlib import Path
from flux.distill_diagnostics import safe_code,transfer_evidence,monitored_download

class DiagnosticsTests(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(dir='/workspace',prefix='distill-diagnostics-'))
  self.src=self.root/'private';self.src.mkdir();self.out=self.root/'input-diagnostics'
  self.control={'url':'https://storage.googleapis.com/fixture?signature=DO_NOT_PERSIST','bytes':10,'sha256':'a'*64}
  self.events=[]
 def emit(self,event,**kw):self.events.append(dict(event=event,**kw))
 def test_safe_failure_codes(self):
  for code in ['input_hash_mismatch','download_deadline_exceeded','object_changed','http_403']:
   self.assertEqual(safe_code(code),code)
  self.assertEqual(safe_code(self.control['url']),'unclassified')
  self.assertEqual(safe_code({'secret':'value'}),'unclassified')
 def test_success_evidence_retained(self):
  def download(url,path,size,sha,diag):
   diag.write_text(json.dumps({'published_verified':True,'expected_bytes':size,'failure':None,'url':url,'task_retries':0}))
   return path
  result=monitored_download('teacher-0.pt',self.control,self.src,self.out,download,self.emit)
  self.assertEqual(result,self.src/'teacher-0.pt')
  row=json.loads((self.out/'teacher-0.pt.json').read_text())
  self.assertTrue(row['published_verified']);self.assertEqual(row['task_retries'],0)
  self.assertNotIn('DO_NOT_PERSIST',json.dumps(self.events)+json.dumps(row))
 def test_failure_preserves_original_and_code(self):
  original=RuntimeError(self.control['url'])
  def fail(url,path,size,sha,diag):
   diag.write_text(json.dumps({'published_verified':False,'failure':'input_hash_mismatch','url':url}))
   raise original
  with self.assertRaises(RuntimeError) as got:
   monitored_download('dataset-0.tar.gz',self.control,self.src,self.out,fail,self.emit)
  self.assertIs(got.exception,original)
  self.assertEqual(self.events[-1]['failure_code'],'input_hash_mismatch')
  self.assertFalse(self.events[-1]['published_verified'])
  self.assertNotIn('DO_NOT_PERSIST',json.dumps(self.events)+(self.out/'dataset-0.tar.gz.json').read_text())
 def test_missing_diagnostics_explicit(self):
  def fail(*args):raise TimeoutError('no response')
  with self.assertRaises(TimeoutError):monitored_download('teacher-1.pt',self.control,self.src,self.out,fail,self.emit)
  self.assertFalse(self.events[-1]['diagnostics_available'])
 def test_invalid_name_rejected(self):
  with self.assertRaises(ValueError):transfer_evidence('../secret',self.src/'anything',self.root/'out.json')
 def test_unknown_failure_text_never_retained(self):
  p=self.src/'input.json';p.write_text(json.dumps({'failure':self.control['url'],'url':self.control['url']}))
  row=transfer_evidence('teacher-0.pt',p,self.root/'out.json')
  self.assertEqual(row['failure_code'],'unclassified');self.assertNotIn('DO_NOT_PERSIST',json.dumps(row))
 def test_oversize_and_symlink_not_followed(self):
  p=self.src/'large.json';p.write_text('x'*(128*1024+1));link=self.src/'linked.json';link.symlink_to(p)
  for source in [p,link]:
   row=transfer_evidence('teacher-0.pt',source,self.root/'out.json')
   self.assertFalse(row['diagnostics_available'])

if __name__=='__main__':unittest.main()