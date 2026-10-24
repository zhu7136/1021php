"""CPU-only startup diagnostics, including an actual native watchdog expiry."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

FLUX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLUX))
import startup_diagnostics as diag


class StartupDiagnosticsTests(unittest.TestCase):
    def events(self, output):
        return [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]

    def test_launcher_success_keeps_return_and_cancels_watchdog(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(diag.faulthandler, 'dump_traceback_later') as arm, \
             patch.object(diag.faulthandler, 'cancel_dump_traceback_later') as cancel:
            self.assertEqual(diag.bounded_launcher(lambda x:x+1)(2), 3)
        self.assertEqual(arm.call_args.args[0], 300)
        self.assertTrue(arm.call_args.kwargs['exit'])
        cancel.assert_called_once()
        self.assertEqual(self.events(output)[-1]['event'], 'diagnostic_phase_complete')

    def test_launcher_failure_propagates_without_retry(self):
        calls = []
        def fail():
            calls.append(1)
            raise ValueError('fixture')
        with contextlib.redirect_stdout(io.StringIO()), patch.object(diag.faulthandler, 'dump_traceback_later'), \
             patch.object(diag.faulthandler, 'cancel_dump_traceback_later') as cancel:
            with self.assertRaises(ValueError):
                diag.bounded_launcher(fail)()
        self.assertEqual(len(calls), 1)
        cancel.assert_called_once()

    def test_native_watchdog_exits_child_and_reports_stack(self):
        code = ("import sys,time;sys.path.insert(0,"+repr(str(FLUX))+");"
                "import startup_diagnostics as d;d.bounded_launcher(lambda:time.sleep(10),.15)()")
        child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=4)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn('Timeout', child.stderr)
        self.assertIn('initialization_deadline_armed', child.stdout)

    def test_class_methods_keep_values_and_only_trace_first_step(self):
        class Env:
            def __init__(self, value):self.value=value
            def reset_all(self):return self.value
            def step(self, x):return x+1
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            diag.instrument_class(Env)
            env=Env(7)
            self.assertEqual(env.reset_all(),7)
            self.assertEqual(env.step(1),2)
            self.assertEqual(env.step(2),3)
            diag.instrument_class(Env)
        starts=[e['phase'] for e in self.events(output) if e['event']=='diagnostic_phase_start']
        self.assertEqual(starts.count('Env.step'),1)
        self.assertIn('Env.reset_all',starts)

    def test_checkpoint_claim_requires_existing_file(self):
        class Algo:
            def save(self, path):return None
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            diag.instrument_class(Algo)
            Algo().save('/workspace/no-such-checkpoint-diagnostic.pt')
        event=next(e for e in self.events(output) if e['event']=='checkpoint_save_returned')
        self.assertFalse(event['file_exists'])
        self.assertEqual(event['bytes'],0)

    def test_nested_phase_restores_after_exception(self):
        before=diag._phase
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            with diag.phase('outer'):
                with diag.phase('inner'):raise RuntimeError('fixture')
        self.assertEqual(diag._phase,before)

    def test_log_mirror_incremental_and_redacts(self):
        scratch=Path('/workspace/hud04_startup_diagnostic_tests');scratch.mkdir(exist_ok=True)
        root=Path(tempfile.mkdtemp(dir=scratch))
        path=root/'kit.log';path.write_text('token=fixture-private\nhttps://example.org/?key=private\n')
        mirror=diag.LogMirror(root,0)
        output=io.StringIO()
        with patch.object(mirror,'paths',return_value=[path]),contextlib.redirect_stdout(output):
            mirror.poll();mirror.poll()
        events=self.events(output)
        self.assertEqual(len(events),1)
        self.assertNotIn('fixture-private',output.getvalue())
        self.assertNotIn('example.org',output.getvalue())

    def test_parent_wait_returns_child_exit_code(self):
        child=subprocess.Popen([sys.executable,'-c','pass'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diag.wait_for_child(child,Path('/workspace/no-job-logs'),3,.1),0)

    def test_parent_wait_deadline_does_not_restart_child(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])
        try:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.TimeoutExpired):
                diag.wait_for_child(child,Path('/workspace/no-job-logs'),.1,.05)
        finally:
            child.terminate();child.wait(timeout=3)


if __name__=='__main__':unittest.main()