"""CPU-only contracts/fixtures; actual CUDA gate remains mandatory per task."""
import os,json,unittest
from pathlib import Path
os.environ.setdefault('PHP_G1_CLIMB_KIND','low')
from flux import g1_climb_contract as c
source=c.adapt(Path(__file__).with_name('test_g1_lowstep.py').read_text())
source=source.replace("'formal_started_monotonic':500","'training_started_monotonic':500")
source=source.replace("if __name__=='__main__':unittest.main()",'')
exec(compile(source,__file__,'exec'),globals())

class ClimbSpecific(unittest.TestCase):
    def test_manifest_and_isolation(self):
        d=json.loads(Path(__file__).parents[1].joinpath('release-assets.json').read_text())['components']['motions-'+c.kind()+'-climb-76']
        self.assertEqual(d['motion_count'],40);self.assertEqual(len(d['files']),81)
        for builder in [c.worker_source,c.bootstrap_source,c.teacher_source,c.input_source,c.algo_source]:
            s=builder(Path(__file__).parent);compile(s,'generated_climb','exec')
            self.assertNotIn('motions-low-step',s);self.assertNotIn('motions-high-step',s)
            self.assertNotIn('flux.g1_lowstep',s)
        self.assertIn('files=81,motions=40',c.worker_source(Path(__file__).parent))
    def test_total_budget_includes_gate(self):
        self.assertEqual(c.budget_times(1080,{'passed':True,'training_started_monotonic':100,'formal_started_monotonic':145}),(1780,1900))
    def test_wrong_dataset(self):
        with self.assertRaises(ValueError):c.validate_scope({'dataset':'motions-'+('high' if c.kind()=='low' else 'low')+'-climb-76'})
    def test_actual_worker_control_decoder(self):
        from flux.control_codec import encode_control
        ns={'__file__':str(Path(__file__).with_name('run_smoke.py')),'__name__':'cpu_only'}
        exec(compile(c.worker_source(Path(__file__).parent),'worker','exec'),ns)
        for obj in [{},{'url':'https://storage.googleapis.com/example','sha256':'a'*64,'bytes':1},{'turn_id':'cpu'}]:
            self.assertEqual(ns['decode_control'](encode_control(obj)),obj)

if __name__=='__main__':unittest.main()