"""CPU contracts, masking gradients, native-method fixture; no simulator/GPU."""
import ast,copy,io,json,os,tempfile,tarfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from flux import five_distill_contract as c

def scope():return dict(gpu_start_authorized=True,trial_count=1,automatic_retries=0,automatic_continuation=False,
 worker_seconds_max=1800,gpu='1x4090D24GB',num_envs=80,teachers_frozen=True,student_count=1,mode='masked_pure_dagger',
 teacher_warmup_updates=5,num_steps_per_env=24,num_learning_epochs=2,num_mini_batches=8,skills=list(c.SKILLS),
 code_commit='a'*40,user_approval_utc='2026-10-27T07:50:12Z')

class ContractTests(unittest.TestCase):
 def test_scope(self):self.assertEqual(c.validate_scope(scope())['worker_seconds_max'],1800)
 def test_scope_mutations(self):
  for k,v in [('trial_count',2),('worker_seconds_max',1801),('num_envs',4096),('teachers_frozen',False),('student_count',5),('automatic_retries',1),('mode','ppo'),('gpu','2x4090D24GB')]:
   with self.subTest(k=k),self.assertRaises(ValueError):c.validate_scope(dict(scope(),**{k:v}))
 def test_teacher_order(self):
  s=scope();s['skills']=list(reversed(s['skills']))
  with self.assertRaises(ValueError):c.validate_scope(s)
 def test_teacher_hashes(self):
  self.assertEqual(len(c.TEACHERS),5);self.assertEqual([r[2] for r in c.TEACHERS],[4846,2594,2629,1345,1323])
  self.assertEqual(len({r[4] for r in c.TEACHERS}),5)
 def test_data_size_bound(self):
  self.assertEqual(len(c.DATA),4);self.assertTrue(all(r[2]<100*1024**2 for r in c.DATA))
 def test_masked_loss_gradient(self):
  x=torch.tensor([[2.,2.],[9.,9.]],requires_grad=True);y=torch.tensor([[1.,1.],[0.,0.]])
  loss=c.masked_loss(x,y);self.assertEqual(loss.item(),1);loss.backward()
  self.assertTrue(torch.equal(x.grad[1],torch.zeros(2)));self.assertTrue((x.grad[0]!=0).all())
 def test_all_invalid_finite_graph(self):
  x=torch.ones(8,29,requires_grad=True);loss=c.masked_loss(x,torch.zeros_like(x));loss.backward()
  self.assertEqual(loss.item(),0);self.assertEqual(x.grad.abs().sum().item(),0)
 def test_valid_normalization(self):
  x=torch.ones(10,29);y=torch.zeros_like(x);y[0]=2
  self.assertEqual(c.masked_loss(x,y).item(),1.)
 def test_archive_extra_or_link(self):
  for link in [True,False]:
   root=Path(tempfile.mkdtemp(dir='/workspace',prefix='five-distill-test-'));f=root/'bad.tar.gz'
   with tarfile.open(f,'x:gz') as tar:
    m=tarfile.TarInfo('../bad' if not link else 'expected');m.type=tarfile.SYMTYPE if link else tarfile.REGTYPE;m.linkname='/bad';tar.addfile(m)
   with self.assertRaises(ValueError):c.unpack_data(f,root/'output',{'files':[{'path':'expected','size':0,'sha256':'a'*64}]})
 def test_compile_sources(self):
  for p in Path(__file__).parent.glob('*five_distill*.py'):compile(p.read_text(),str(p),'exec')
 def test_worker_budgets_and_no_fallback(self):
  s=Path(__file__).with_name('run_five_distill.py').read_text()
  self.assertIn('1798-',s);self.assertIn('signal.alarm(1600)',s);self.assertIn('time.monotonic()-started>1050',s)
  self.assertNotIn('parallel_download.MAX',s);self.assertNotIn('wandb.init(',s)
 def test_plan_does_not_persist_urls(self):
  s=Path(__file__).with_name('run_five_distill.py').read_text();node=ast.parse(s)
  writes=[ast.get_source_segment(s,n) for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text']
  self.assertTrue(any("'teachers':[str(x)" in w for w in writes))
  self.assertTrue(all("controls" not in w and 'upload_url' not in w for w in writes))
 def test_control_size_roundtrip(self):
  from flux.control_codec import encode_control,decode_control
  for n in [4,5]:
   v={'teachers':[{'url':'https://storage.googleapis.com/fixture?'+('x'*1000),'bytes':10346235,'sha256':'a'*64} for _ in range(n)]}
   self.assertEqual(decode_control(encode_control(v)),v)

if __name__=='__main__':unittest.main()