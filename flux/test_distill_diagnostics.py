"""CPU fault injection for sanitized, retained transport diagnostics."""
import json,tempfile,unittest
from pathlib import Path
from flux.distill_diagnostics import safe_code,transfer_evidence,monitored_download,validate_input_name

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
 def test_source_student_success_and_redaction(self):
  def download(url,path,size,sha,diag):
   diag.write_text(json.dumps({'published_verified':True,'expected_bytes':size,'failure':None,'url':url}))
   return path
  result=monitored_download('source-student.pt',self.control,self.src,self.out,download,self.emit)
  self.assertEqual(result,self.src/'source-student.pt')
  text=(self.out/'source-student.pt.json').read_text()
  self.assertTrue(json.loads(text)['published_verified']);self.assertNotIn('DO_NOT_PERSIST',text+json.dumps(self.events))
 def test_source_student_failure_preserves_exception(self):
  original=TimeoutError('fixture')
  def fail(url,path,size,sha,diag):
   diag.write_text(json.dumps({'published_verified':False,'failure':'download_deadline_exceeded'}));raise original
  with self.assertRaises(TimeoutError) as got:monitored_download('source-student.pt',self.control,self.src,self.out,fail,self.emit)
  self.assertIs(got.exception,original);self.assertEqual(self.events[-1]['failure_code'],'download_deadline_exceeded')
 def test_narrow_student_allowlist(self):
  self.assertEqual(validate_input_name('source-student.pt'),'source-student.pt')
  for name in ['source-student.pt.json','source-student2.pt','../source-student.pt','/source-student.pt','teacher-5.pt','dataset-4.tar.gz',None]:
   with self.subTest(name=name),self.assertRaises(ValueError):validate_input_name(name)
 def test_invalid_name_rejected_before_transfer(self):
  def forbidden(*args):self.fail('Transfer must not begin for rejected name')
  with self.assertRaises(ValueError):monitored_download('../source-student.pt',self.control,self.src,self.out,forbidden,self.emit)
  self.assertEqual(self.events,[])
 def test_all_video_transfer_names_preflight(self):
  from flux.student_video_contract import input_jobs
  jobs=input_jobs([self.control]*4,self.control)
  self.assertEqual([n for n,_ in jobs],['dataset-0.tar.gz','dataset-1.tar.gz','dataset-2.tar.gz','dataset-3.tar.gz','source-student.pt'])
  for name,_ in jobs:
   row=transfer_evidence(name,self.src/'absent.json',self.out/(name+'.json'))
   self.assertFalse(row['diagnostics_available'])
  with self.assertRaises(ValueError):input_jobs([self.control]*5,self.control)
 def test_classified_chunk_evidence_survives_sanitizing(self):
  p=self.src/'diag.json';p.write_text(json.dumps({'failure':'socket_timeout','failure_stage':'range_read',
   'url':self.control['url'],'chunks':[{'index':2,'start':16,'end':23,'bytes_received':3,
   'outcome':'failed','failure':'socket_timeout','stage':'range_read','url':self.control['url']}]}))
  row=transfer_evidence('source-student.pt',p,self.out/'safe.json')
  self.assertEqual(row['failure_code'],'socket_timeout');self.assertEqual(row['failure_stage'],'range_read')
  self.assertEqual(row['failed_chunks'][0]['bytes_received'],3)
  self.assertNotIn('DO_NOT_PERSIST',json.dumps(row))
 def test_untrusted_chunk_fields_are_bounded_and_redacted(self):
  p=self.src/'diag.json';p.write_text(json.dumps({'failure':self.control['url'],'failure_stage':self.control['url'],
   'chunks':[{'index':True,'start':-1,'bytes_received':10**20,'end':self.control['url'],
   'outcome':'failed','failure':self.control['url'],'stage':self.control['url']}]*100}))
  row=transfer_evidence('source-student.pt',p,self.out/'safe.json')
  self.assertNotIn('failure_stage',row);self.assertEqual(len(row['failed_chunks']),8)
  self.assertTrue(all(c=={'failure_code':'unclassified'} for c in row['failed_chunks']))
  self.assertNotIn('DO_NOT_PERSIST',json.dumps(row))
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