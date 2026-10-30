"""CPU-only log recovery and exception redaction regression tests."""
import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from hud04_diagnostic_logging import emit_action_probe_diagnostics, diagnostic_error


class LoggingTests(unittest.TestCase):
    def record(self, step=1):
        return {'schema':'hud04-same-state-action-probe-v1','step':step,
                'teacher_actions_applied':False,'route':-1,'action_rmse':None}

    def test_case_identity_and_returncode_survive_recovery(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder)
            (root/'same-state-actions.jsonl').write_text(json.dumps(self.record())+'\n')
            row={'skill':'high-step'};code=1
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                emit_action_probe_diagnostics(root)
            record={'skill':row['skill'],'native_returncode':code}
            self.assertEqual(record,{'skill':'high-step','native_returncode':1})
            self.assertEqual(json.loads(stdout.getvalue())['route'],-1)

    def test_worker_has_no_row_reassignment_in_case_loop(self):
        tree=ast.parse(Path(__file__).with_name('obstacle_video_worker.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets)
                             for n in ast.walk(tree)))
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                            and n.func.id=='emit_action_probe_diagnostics' for n in ast.walk(tree)))

    def test_bad_records_rejected_before_any_action_row_emitted(self):
        for bad in [self.record(3),{**self.record(),'teacher_actions_applied':True},
                    {**self.record(),'schema':'wrong'},{**self.record(),'action_rmse':float('nan')}]:
            with self.subTest(bad=bad),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                root=Path(folder);(root/'same-state-actions.jsonl').write_text(json.dumps(bad)+'\n')
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    with self.assertRaises(ValueError):emit_action_probe_diagnostics(root)
                self.assertEqual(stdout.getvalue(),'')

    def test_row_limit_enforced(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder)
            (root/'same-state-actions.jsonl').write_text('\n'.join(json.dumps(self.record(i)) for i in range(1,1002)))
            with self.assertRaises(ValueError):emit_action_probe_diagnostics(root)

    def test_missing_files_are_noop(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:emit_action_probe_diagnostics(folder)
            self.assertEqual(stdout.getvalue(),'')

    def test_known_exception_has_message_and_stack(self):
        try:raise ValueError('Teacher observation route mismatch')
        except ValueError as exc:record=diagnostic_error(exc,'same_state_action_probe',86)
        self.assertFalse(record['message_withheld'])
        self.assertEqual(record['attempted_policy_step'],86)
        self.assertEqual(record['frames'][-1]['file'],Path(__file__).name)
        self.assertNotIn('/workspace',json.dumps(record))

    def test_unknown_exception_does_not_leak_capability_or_locals(self):
        marker='https://example.invalid/?signature=SYNTHETIC_NOT_A_CREDENTIAL'
        try:raise RuntimeError(marker)
        except RuntimeError as exc:record=diagnostic_error(exc,'environment_step',1)
        self.assertTrue(record['message_withheld'])
        self.assertNotIn(marker,json.dumps(record))
        self.assertNotIn('SYNTHETIC_NOT_A_CREDENTIAL',json.dumps(record))


if __name__=='__main__':unittest.main()