"""CPU supervisor tests; no GPU or training evidence is generated."""
import importlib.util,json,os,subprocess,sys,tempfile,time,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from three_teacher_parent import extract_data,validate_result,INPUTS
from worker_utils import stage,ACTIVE
from bootstrap_base import clean_env
from scoped_transport import digest

class WorkerTests(unittest.TestCase):
 def test_dry_run_and_syntax(self):
  p=subprocess.run([sys.executable,str(ROOT/'source/three_teacher_parent.py')],capture_output=True,text=True,timeout=10)
  self.assertEqual(p.returncode,0);r=json.loads(p.stdout)
  self.assertFalse(r['gpu_started']);self.assertEqual((r['num_envs'],r['max_updates'],r['automatic_retries']),(6,5,0))
  for f in (ROOT/'source').glob('*.py'):compile(f.read_text(),str(f),'exec')
 def test_sanitized_child_environment(self):
  os.environ['HUD04_TEST_SECRET']='not-a-real-secret'
  try:
   e=clean_env();self.assertNotIn('HUD04_TEST_SECRET',e)
   for k in ('flux_api2','github','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY'):self.assertNotIn(k,e)
  finally:os.environ.pop('HUD04_TEST_SECRET')
 def test_process_deadline(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as d:
   with self.assertRaises(TimeoutError):stage([sys.executable,'-c','import time;time.sleep(5)'],Path(d)/'log',clean_env(),time.monotonic()+.15)
   self.assertFalse(ACTIVE)
 def test_data_zip_identity_and_escape(self):
  p=Path('/workspace/three-teacher-adaptation/hud04-three-skill-data.zip')
  self.assertEqual((p.stat().st_size,digest(p)),INPUTS['data'])
  with tempfile.TemporaryDirectory(dir=ROOT) as d:
   z=Path(d)/'bad.zip'
   with zipfile.ZipFile(z,'w') as a:a.writestr('data/../../escape','x')
   with self.assertRaises(ValueError):extract_data(z,Path(d)/'out')
 def test_native_failure_never_masked(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as d:
   with self.assertRaises(ValueError):validate_result(d,1,'TASK_20261031_999')

if __name__=='__main__':unittest.main(verbosity=2)