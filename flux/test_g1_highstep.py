"""CPU regression only, not CUDA/simulator evidence."""
import unittest,json
from pathlib import Path
from flux import g1_highstep_contract as c

# Reuse all low-step scope and actual checkpoint-corruption fixtures on the
# high-step implementation, with every fixture target adapted as well.
source=c.adapt(Path(__file__).with_name('test_g1_lowstep.py').read_text())
source=source.replace("if __name__=='__main__':unittest.main()",'')
exec(compile(source,__file__,'exec'),globals())

class HighSpecific(unittest.TestCase):
    def test_manifest(self):
        d=json.loads(Path(__file__).parents[1].joinpath('release-assets.json').read_text())['components']['motions-high-step']
        self.assertEqual(d['motion_count'],60)
        self.assertEqual(len(d['files']),121)
        self.assertEqual(d['archive']['sha256'],'c4cf035cf2a81336a42277584ef65ee16044f9c5f9d921b08e84c1adaab7381d')
    def test_no_low_dataset_or_entry_in_generated_source(self):
        for build in [c.worker_source,c.teacher_source,c.bootstrap_source]:
            text=build(Path(__file__).parent)
            self.assertNotIn('motions-low-step',text)
            self.assertNotIn('flux.g1_lowstep',text)
        self.assertIn('files=121,motions=60',c.worker_source(Path(__file__).parent))
    def test_low_scope_rejected(self):
        with self.assertRaises(ValueError):c.validate_scope({'dataset':'motions-low-step'})
    def test_decoder_actual_worker(self):
        from flux.control_codec import encode_control
        ns={'__file__':str(Path(__file__).with_name('run_smoke.py')),'__name__':'cpu_preflight'}
        exec(compile(c.worker_source(Path(__file__).parent),'generated_worker','exec'),ns)
        for obj in [{},{'thread_id':'cpu','turn_id':'test'},{'url':'https://storage.googleapis.com/a','sha256':'a'*64,'bytes':1}]:
            self.assertEqual(ns['decode_control'](encode_control(obj)),obj)
        self.assertIn("--highstep-input-b64",c.worker_source(Path(__file__).parent))

if __name__=='__main__':unittest.main()