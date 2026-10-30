import contextlib,io,json,math,tempfile,unittest
from pathlib import Path
from hud04_compact_action_log import aggregate,compact_events
from hud04_diagnostic_logging import emit_action_probe_diagnostics

def row(step,valid=True):
    return {'schema':'hud04-same-state-action-probe-v1','step':step,'reference_frame':step-1,
      'teacher_actions_applied':False,'comparison_valid':valid,'route':2 if valid else -1,
      'student_actions':[2.]*29,'teacher_actions':[1.]*29 if valid else None,
      'depth_stats':{'rms':.4,'zero_fraction':0.},'actor_stats':{'rms':.6},
      'actor_terms':{'actions':{'rms':.2}},'action_rmse':1. if valid else None,
      'student_action_stats':{'rms':2.},'teacher_action_stats':{'rms':1.} if valid else None}

class CompactTests(unittest.TestCase):
    def test_exact_aggregate(self):
        a=aggregate([row(1),row(2)])
        for key,value in [('action_rmse',1),('relative_rmse',1),('pooled_cosine',1),('student_rms',2),('teacher_rms',1)]:
            self.assertEqual(a[key],value)
        self.assertEqual(a['per_joint_bias'],[1.]*29)
    def test_invalid_expert_excluded(self):
        a=aggregate([row(1),row(2,False)])
        self.assertEqual(a['compared_steps'],1);self.assertEqual(a['expert_invalid_steps'],1)
        self.assertEqual(a['action_rmse'],1.)
    def test_all_masked_no_fake_target(self):
        self.assertNotIn('action_rmse',aggregate([row(1,False)]))
    def test_segments_and_sampling(self):
        events=[json.loads(x) for x in compact_events([row(i) for i in range(1,1001)])]
        self.assertEqual(len(events),101);self.assertEqual(events[0]['step'],1);self.assertEqual(events[-2]['step'],1000)
        summary=events[-1];self.assertEqual(summary['all_rows']['compared_steps'],1000)
        self.assertEqual(sum(s['steps'] for s in summary['segments']),1000)
        self.assertLess(sum(len(json.dumps(r)) for r in events),110000)
    def test_full_rows_not_mutated(self):
        rows=[row(i) for i in range(1,151)];before=json.dumps(rows)
        compact_events(rows);self.assertEqual(json.dumps(rows),before)
    def test_nonfinite_actions_refused(self):
        r=row(1);r['student_actions'][0]=float('nan')
        with self.assertRaises(ValueError):compact_events([r])
    def test_step_gap_refused(self):
        with self.assertRaises(ValueError):compact_events([row(2)])
    def test_compact_recovery_keeps_numeric_summary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            p=Path(folder);(p/'same-state-actions.jsonl').write_text('\n'.join(json.dumps(row(i)) for i in range(1,151)))
            with contextlib.redirect_stdout(io.StringIO()) as out:emit_action_probe_diagnostics(p,compact=True)
            records=[json.loads(l) for l in out.getvalue().splitlines()]
            self.assertEqual(records[-1]['event'],'hud04_compact_action_summary')
            self.assertEqual(records[-1]['full_rows'],150)
            self.assertNotIn('student_actions',out.getvalue())

if __name__=='__main__':unittest.main()