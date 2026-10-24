"""CPU parity and bounded-allocation tests; no simulator or paid compute."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wbt_training.utils.motion_replication import stream_offset_motions_to_file


def legacy(data, origins, offsets):
    """Unmodified arithmetic from obstacle_helpers before this patch."""
    body = torch.tensor(data['body_pos_w'], dtype=torch.float32)
    body[:, :, :2] += torch.as_tensor(offsets, dtype=torch.float32).unsqueeze(1)
    E, T = len(origins), len(body)
    body = body.unsqueeze(0) + torch.tensor(origins, dtype=body.dtype)[:, None, None, :]
    body = body.reshape(E*T, *body.shape[2:])
    result = {}
    for k, v in data.items():
        if k == 'body_pos_w':
            continue
        arr = np.asarray(v)
        result[k] = np.concatenate([arr]*E) if arr.ndim and arr.shape[0] == T else v
    result['body_pos_w'] = body.numpy()
    return result


class MotionReplicationTests(unittest.TestCase):
    def setUp(self):
        parent = Path('/workspace/hud04_replication_tests')
        parent.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=parent))
        rng = np.random.default_rng(41)
        T, B, J = 17, 4, 3
        self.data = {
            'fps': np.array([50.]), 'body_names': np.array(['a','b','c','d']),
            'joint_names': np.array(['x','y','z']),
            'joint_pos': rng.normal(size=(T,J+7)),
            'joint_vel': rng.normal(size=(T,J+6)),
            'body_pos_w': rng.normal(size=(T,B,3)),
            'body_quat_w': rng.normal(size=(T,B,4)),
            'body_lin_vel_w': rng.normal(size=(T,B,3)),
            'body_ang_vel_w': rng.normal(size=(T,B,3)),
            'motion_ends': np.arange(T)%5 == 0,
            'motion_idxs': np.arange(T,dtype=np.int64)//5,
            'other_float64': rng.normal(size=(T,2)),
        }
        self.offsets = rng.normal(size=(T,2)).astype(np.float32)
        self.origins = rng.normal(size=(32,3))

    def run_writer(self, data=None, E=3, **kwargs):
        path = self.root/'output.npz'
        with redirect_stdout(io.StringIO()):
            report = stream_offset_motions_to_file(
                self.data if data is None else data, self.origins[:E],
                self.offsets, path, **kwargs)
        with np.load(path, allow_pickle=False) as out:
            arrays = {k: out[k] for k in out.files}
        return arrays, report

    def test_exact_loader_float32_values_for_1_3_32_variants(self):
        for E in (1,3,32):
            expected = legacy(self.data, self.origins[:E], self.offsets)
            actual, _ = self.run_writer(E=E, chunk_bytes=200)
            self.assertEqual(set(actual), set(expected))
            for k in expected:
                if k in {'joint_pos','joint_vel','body_quat_w','body_lin_vel_w','body_ang_vel_w'}:
                    # Compare the actual pinned loader's conversion, not tolerance.
                    np.testing.assert_array_equal(actual[k], torch.tensor(expected[k], dtype=torch.float32).numpy())
                else:
                    np.testing.assert_array_equal(actual[k], expected[k])

    def test_optional_original_dtypes_and_values_are_preserved(self):
        actual, _ = self.run_writer(narrow_float32=False)
        expected = legacy(self.data, self.origins[:3], self.offsets)
        for k in expected:
            self.assertEqual(actual[k].dtype, expected[k].dtype)
            np.testing.assert_array_equal(actual[k], expected[k])

    def test_one_source_read_per_key(self):
        class Counting(dict):
            def __getitem__(self, key):
                counts[key] = counts.get(key, 0)+1
                return super().__getitem__(key)
            def items(self):
                raise AssertionError('Do not iterate NpzFile.items()')
        counts = {}
        self.run_writer(data=Counting(self.data))
        self.assertEqual(counts, {k:1 for k in self.data})

    def test_no_concatenate_and_source_unchanged(self):
        before = {k:v.copy() for k,v in self.data.items()}
        with patch('numpy.concatenate', side_effect=AssertionError('No expanded concatenation')):
            self.run_writer()
        for k in before:
            np.testing.assert_array_equal(self.data[k], before[k])

    def test_in_place_npz_open_is_safe(self):
        source = self.root/'same.npz'
        np.savez(source, **self.data)
        expected = legacy(self.data, self.origins[:3], self.offsets)
        with np.load(source) as data, redirect_stdout(io.StringIO()):
            stream_offset_motions_to_file(data, self.origins[:3], self.offsets, source)
        with np.load(source) as out:
            np.testing.assert_array_equal(out['body_pos_w'], expected['body_pos_w'])
            np.testing.assert_array_equal(out['motion_ends'], expected['motion_ends'])

    def test_noncontiguous_input_and_tensor_offsets(self):
        self.data['joint_pos'] = np.asfortranarray(self.data['joint_pos'])
        actual, _ = self.run_writer()
        expected = legacy(self.data, self.origins[:3], self.offsets)
        np.testing.assert_array_equal(actual['joint_pos'], expected['joint_pos'].astype(np.float32))
        with redirect_stdout(io.StringIO()):
            stream_offset_motions_to_file(self.data, torch.tensor(self.origins[:2]),
                                          torch.tensor(self.offsets), self.root/'tensor.npz')

    def test_shape_dtype_and_payload_reporting(self):
        actual, report = self.run_writer()
        self.assertEqual(report['payload_bytes'], sum(a.nbytes for a in actual.values()))
        self.assertEqual(actual['motion_ends'].dtype, np.bool_)
        self.assertEqual(actual['motion_idxs'].dtype, np.int64)
        self.assertEqual(actual['other_float64'].dtype, np.float64)
        self.assertEqual(actual['joint_pos'].shape, (51,10))
        self.assertEqual(report['expanded_frames'], 51)

    def test_failure_leaves_existing_destination_unchanged(self):
        destination = self.root/'output.npz'
        destination.write_bytes(b'prior complete output')
        with patch('numpy.lib.format.write_array', side_effect=OSError('injected disk failure')):
            with self.assertRaises(OSError):
                self.run_writer()
        self.assertEqual(destination.read_bytes(), b'prior complete output')

    def test_reject_invalid_dimensions_and_objects(self):
        with redirect_stdout(io.StringIO()):
            for origins in (np.zeros((0,3)), np.zeros((2,2))):
                with self.assertRaises(ValueError):
                    stream_offset_motions_to_file(self.data, origins, self.offsets, self.root/'invalid.npz')
        data = dict(self.data, unknown=np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            self.run_writer(data=data)

    def test_chunked_frame_writes_stay_bounded(self):
        import zipfile
        original = zipfile._ZipWriteFile.write
        lengths = []
        def write(member, data):
            lengths.append(len(data))
            return original(member, data)
        with patch.object(zipfile._ZipWriteFile, 'write', write):
            self.run_writer(chunk_bytes=512)
        # Includes tiny NPY headers and metadata. A frame is at most 128 B.
        self.assertLessEqual(max(lengths), 512)

    def test_requires_body_positions(self):
        data = {k:v for k,v in self.data.items() if k != 'body_pos_w'}
        with self.assertRaisesRegex(ValueError, 'body_pos_w'):
            self.run_writer(data=data)

    def test_header_audit_matches_legacy_and_streamed_payloads(self):
        sys.path.insert(0, str(ROOT/'flux'))
        from audit_motion_replication import audit
        inputs = self.root/'inputs'
        inputs.mkdir()
        data = {k:v for k,v in self.data.items() if k != 'other_float64'}
        np.savez(inputs/'test_motion.npz', **data)
        report = audit(inputs, 3)
        expected = legacy(data, self.origins[:3], self.offsets)
        actual, measured = self.run_writer(data=data)
        self.assertEqual(report['legacy_payload_bytes'],sum(a.nbytes for a in expected.values()))
        self.assertEqual(report['streamed_payload_bytes'],measured['payload_bytes'])
        self.assertEqual(report['source_frames'],17)


if __name__ == '__main__':
    unittest.main()