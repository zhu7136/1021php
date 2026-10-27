"""CPU-only exact restore/optimizer advancement fixtures for each climb source."""
import os,re,json,unittest
from pathlib import Path
os.environ.setdefault('PHP_G1_CLIMB_KIND','low')
from flux.g1_climb_contract import kind
from flux.g1_climb_resume_input import SOURCE_ITERATION,SOURCE_UPDATES
text=Path(__file__).with_name('test_g1_resume224.py').read_text()
for a,b in [('g1_resume224','g1_climb_resume'),('G1Resume224','G1ClimbResume'),('g1_lowstep','g1_climb'),('G1Lowstep','G1Climb'),('motions-low-step','motions-'+kind()+'-climb-76'),('--lowstep-input-b64','--climb-input-b64')]:text=text.replace(a,b)
for old,new in [(16840,SOURCE_UPDATES*20),(841,SOURCE_ITERATION),(842,SOURCE_UPDATES),(863,SOURCE_UPDATES+21),(861,SOURCE_ITERATION+20)]:text=re.sub(r'\b'+str(old)+r'\b',str(new),text)
text=text.replace("if __name__=='__main__':unittest.main()",'')
exec(compile(text,__file__,'exec'),globals())
class ClimbResumeSpecific(unittest.TestCase):
    def test_total_budget(self):
        self.assertEqual(c.budget_times(1080,{'passed':True,'training_started_monotonic':100,'formal_started_monotonic':145}),(1780,1900))
    def test_data_and_source_binding(self):
        self.assertEqual(inp.SOURCE_ITERATION,SOURCE_ITERATION)
        for build in [c.worker_source,c.teacher_source,c.bootstrap_source,c.algo_source]:
            s=build(ROOT);compile(s,'actual_climb_resume','exec')
            self.assertNotIn('motions-low-step',s);self.assertNotIn('motions-high-step',s)
        self.assertIn(inp.SOURCE_RUN,c.teacher_source(ROOT))
if __name__=='__main__':unittest.main()