"""CPU regression tests for gm-run control argument forwarding; no GPU."""
import base64
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux import run_smoke


def encode(value):
    return base64.b64encode(json.dumps(value,ensure_ascii=False).encode()).decode()


class SmokeControlTests(unittest.TestCase):
    def test_roundtrip_special_characters(self):
        value={'thread_id':'a with spaces "quotes" 中文','turn_id':'t/1'}
        self.assertEqual(run_smoke.decode_control(encode(value)),value)

    def test_shell_and_plain_split_agree(self):
        args=['gm-run','1021php/flux/run_smoke.py','--aria-provenance-b64',encode({'thread_id':'a b','turn_id':'123'})]
        command=shlex.join(args)
        self.assertEqual(command.split(),args)
        self.assertEqual(shlex.split(command),args)

    def test_invalid_encoding_rejected(self):
        with self.assertRaises(ValueError):
            run_smoke.decode_control('not!base64')

    def test_non_object_rejected(self):
        with self.assertRaises(ValueError):
            run_smoke.decode_control(encode(['not-an-object']))

    def test_oversized_rejected(self):
        with self.assertRaises(ValueError):
            run_smoke.decode_control(encode({'x':'y'*9000}))

    def test_bad_control_fails_before_preparation(self):
        parent=Path(tempfile.mkdtemp(prefix='php-smoke-control-',dir='/workspace'))
        output=parent/'not-created'
        argv=['run_smoke.py','--task-id','TASK_20261025_019','--expected-commit','fake','--upload-url','fake','--probe-url','fake',
              '--aria-provenance-b64','invalid!']
        with patch.object(run_smoke,'OUT',output),patch.object(sys,'argv',argv),patch.object(run_smoke,'launcher') as launcher:
            with self.assertRaises(ValueError):
                run_smoke.main()
            launcher.assert_not_called()
        self.assertFalse(output.exists())

    def test_new_task_output_is_separate(self):
        self.assertNotEqual(run_smoke.task_output('TASK_20261025_019'),run_smoke.task_output('TASK_20261025_018'))

    def test_task_output_stays_in_outputs(self):
        self.assertEqual(run_smoke.task_output('TASK_20261025_019').parent,run_smoke.ROOT/'outputs')

    def test_invalid_task_output_rejected(self):
        for value in ('../old','TASK_20261025_019/../old','TASK_20261025_019;echo',''):
            with self.assertRaises(ValueError):
                run_smoke.task_output(value)


if __name__=='__main__':
    unittest.main()