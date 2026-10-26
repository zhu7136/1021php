import ast,copy,json,unittest
from pathlib import Path
from flux import g1_eval283_protocol as p

class FrozenEvalTests(unittest.TestCase):
    def scope(self):
        return {'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,'gpu':'1x4090D-24G',
                'worker_seconds_max':1800,'training_updates':0,'source_run':p.SOURCE_RUN,
                'checkpoint_sha256':p.CHECKPOINT_SHA256,'dataset':'motions-low-step','pairs':56,
                'terrain_variants':64,'repeats':3,'seeds':[42,43,44],
                'output_upload_probe_passed':True,'code_commit':'a'*40}
    def rows(self):return [{'clip_index':i,'seed':s,'status':'not_tested'} for s in p.SEEDS for i in range(56)]
    def test_valid_scope(self):self.assertEqual(p.validate_scope(self.scope())['training_updates'],0)
    def test_training_budget_and_wrong_checkpoint_fail(self):
        for key,value in [('training_updates',1),('worker_seconds_max',1801),('trial_count',2),
                          ('automatic_retries',1),('source_run','wj7an49u'),('dataset','motions-locomotion'),
                          ('terrain_variants',1),('worker_seconds_max',True),('gpu_start_authorized',False)]:
            s=self.scope();s[key]=value
            with self.assertRaises(ValueError,msg=key):p.validate_scope(s)
    def test_failure_precedes_timeout_and_completion(self):self.assertEqual(p.classify(True,True,True),'failed')
    def test_timeout_never_success(self):self.assertEqual(p.classify(False,True,True),'truncated')
    def test_real_complete(self):self.assertEqual(p.classify(False,False,True),'completed')
    def test_running(self):self.assertIsNone(p.classify(False,False,False))
    def test_untested_explicit_denominator(self):
        rows=self.rows();rows[0]['status']='completed';rows[1]['status']='failed';rows[2]['status']='truncated'
        r=p.summarize(rows);self.assertEqual(r['not_tested'],165);self.assertEqual(r['completion_rate_resolved_only'],.5)
        self.assertEqual(r['completion_fraction_of_planned'],1/168);self.assertFalse(r['full_coverage'])
    def test_duplicates_and_unknown_status_rejected(self):
        rows=self.rows();rows[0]=rows[1]
        with self.assertRaises(ValueError):p.summarize(rows)
        rows=self.rows();rows[0]['status']='success_from_timeout'
        with self.assertRaises(ValueError):p.summarize(rows)
    def test_reset_assignment_only(self):
        fixture='class MotionCommand:\n    def reset(self, env_ids):\n        self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)\n        return 5\n'
        out=p.scheduled_reset_source(fixture);ast.parse(out)
        self.assertIn('self.eval_motion_ids[env_ids]',out);self.assertIn('return 5',out)
        with self.assertRaises(ValueError):p.scheduled_reset_source(fixture.replace('torch.randint','torch.rand'))
    def test_isolated_lineage(self):
        from flux import g1_eval283_input as new,g1_resume276_input as old
        self.assertEqual(new.SOURCE_RUN,p.SOURCE_RUN);self.assertEqual(new.CHECKPOINT_SHA256,p.CHECKPOINT_SHA256)
        self.assertEqual(old.SOURCE_RUN,'wj7an49u');self.assertNotEqual(old.CHECKPOINT_SHA256,new.CHECKPOINT_SHA256)
        with self.assertRaises(ValueError):new.validate_control({'url':'https://storage.googleapis.com/x',
            'sha256':old.CHECKPOINT_SHA256,'bytes':new.CHECKPOINT_BYTES,'source_task':old.SOURCE_TASK,'source_run':old.SOURCE_RUN})
    def test_no_training_calls_in_teacher(self):
        root=Path(__file__).parent
        tree=ast.parse((root/'g1_eval283_teacher.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('learn','backward') for n in ast.walk(tree)))
        for name in ('g1_eval283_teacher.py','run_g1_eval283.py'):ast.parse((root/name).read_text())

if __name__=='__main__':unittest.main()