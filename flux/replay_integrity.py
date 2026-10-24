"""CPU-only encoder preflight and fail-closed replay completion evidence.

The parent verifier intentionally uses only the standard library: its Python
runtime need not be ABI-compatible with Isaac's NumPy/OpenCV runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_mp4(path, width=640, height=360, expected_frames=None):
    import cv2
    capture = cv2.VideoCapture(str(path))
    frames = 0
    try:
        if not capture.isOpened():
            raise RuntimeError('Video cannot be opened')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise RuntimeError('Video frame dimensions are incorrect')
            frames += 1
            if frames > 1000:
                raise RuntimeError('Video exceeds finite replay frame budget')
    finally:
        capture.release()
    if frames == 0 or (expected_frames is not None and frames != expected_frames):
        raise RuntimeError('Video frame count is incorrect or zero')
    return dict(frames=frames, width=width, height=height)


def preflight_mp4(create_video, directory):
    """Exercise the exact Holosoma encode path without ffmpeg or a GPU.

    Test both one-frame early termination and a multi-frame episode. MPEG-4
    Part 2 is inspectable/downloadable but less browser-compatible than H.264.
    """
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for count in (1, 3):
        frames = np.zeros((count, 360, 640, 3), dtype=np.uint8)
        frames[:, 40:120, 60:240, 1] = 180
        frames[:, :, :, 0] = 40
        path = create_video(frames, fps=50, save_dir=directory,
                            output_format='mp4', wandb_logging=False, episode_id=count)
        if path is None or not Path(path).is_file():
            raise RuntimeError('Video encoder produced no file')
        results.append(inspect_mp4(path, expected_frames=count))
    return dict(output_format='mp4', codec='mp4v', ffmpeg_required=False,
                browser_compatibility='limited; download or decode with OpenCV', tests=results)


def _safe_file(out, name):
    relative = Path(name)
    if not isinstance(name, str) or relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Unsafe replay artifact path')
    path = out / relative
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise RuntimeError('Replay artifact symlinks are forbidden')
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError('Required replay artifact is missing or empty')
    return path


def _summary(out, checkpoint_sha256):
    summary = json.loads(_safe_file(out, 'summary.json').read_text())
    expected = dict(steps=1000, policy_steps=999, initialization_zero_action_steps=1,
                    num_envs=1, training_updates=0, terrain_variants=32,
                    checkpoint_sha256=checkpoint_sha256)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError('Replay summary invariant failed: ' + key)
    for key in ('checkpoint_unchanged', 'actor_weights_unchanged',
                'joint_mapping_runtime_verified', 'all_states_and_actions_finite'):
        if summary.get(key) is not True:
            raise RuntimeError('Replay summary invariant failed: ' + key)
    videos = summary.get('videos')
    if not isinstance(videos, list) or not videos or len(videos) > 1001 or len(set(videos)) != len(videos):
        raise RuntimeError('Replay must include unique recorded videos')
    for name in videos:
        path = _safe_file(out, name)
        if Path(name).parts[0] != 'videos' or path.suffix != '.mp4' or 'temp' in path.name:
            raise RuntimeError('Replay requires finalized MP4s, not temporary files')
    return summary


def complete_replay(out, checkpoint_sha256):
    """Run in Isaac's child before shutdown; seal artifacts only after validation."""
    import numpy as np
    out = Path(out)
    summary = _summary(out, checkpoint_sha256)
    bodies = summary['body_names']
    shapes = dict(root_pos=(1000, 3), root_quat_xyzw=(1000, 4),
        body_pos=(1000, len(bodies), 3), joint_pos=(1000, 29), joint_vel=(1000, 29),
        contact_force_norm=(1000, len(bodies)), root_tilt_rad=(1000,),
        root_clearance_proxy_m=(1000,), reset=(1000,), timeout=(1000,), actions=(1000, 29))
    with np.load(_safe_file(out, 'trajectory.npz'), allow_pickle=False) as arrays:
        if set(arrays.files) != set(shapes):
            raise RuntimeError('Trajectory columns are incomplete')
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise RuntimeError('Invalid trajectory shape or nonfinite values: ' + key)
        if np.any(arrays['actions'][0] != 0):
            raise RuntimeError('Initialization action must be zero')
        if int(arrays['reset'].sum()) != summary['resets'] or int(arrays['timeout'].sum()) != summary['timeouts']:
            raise RuntimeError('Reset/timeout summary does not match trajectory')
        action_max = float(np.abs(arrays['actions']).max())
        if not np.isclose(action_max, summary['action_abs_max']):
            raise RuntimeError('Action summary does not match trajectory')
        reset_steps = np.flatnonzero(arrays['reset']).tolist()
    mapping = json.loads(_safe_file(out, 'joint_mapping.json').read_text())
    if (len(mapping) != 29 or [item['policy_index'] for item in mapping] != list(range(29))
            or len({item['joint'] for item in mapping}) != 29
            or len({item['isaac_index'] for item in mapping}) != 29):
        raise RuntimeError('Joint mapping artifact is incomplete')
    event_path = out / 'events.jsonl'
    if not event_path.is_file() or event_path.is_symlink():
        raise RuntimeError('Missing replay events')
    events = [json.loads(line) for line in event_path.read_text().splitlines() if line]
    if [item['step'] for item in events if item.get('event') == 'pre_reset'] != reset_steps:
        raise RuntimeError('Termination events do not match captured resets')
    video_info = {name: inspect_mp4(_safe_file(out, name)) for name in summary['videos']}
    required = ['summary.json', 'trajectory.npz', 'joint_mapping.json', 'events.jsonl',
                'replay_config.yaml', 'encoder_preflight.json', *summary['videos']]
    artifacts = {}
    for name in required:
        path = out / name
        # Zero-byte events are valid when no environment resets occurred.
        if not path.is_file() or path.is_symlink() or (path.stat().st_size == 0 and name != 'events.jsonl'):
            raise RuntimeError('Missing completion artifact: ' + name)
        artifacts[name] = dict(bytes=path.stat().st_size, sha256=file_digest(path))
    manifest = dict(version=1, steps=1000, policy_steps=999, training_updates=0,
                    checkpoint_sha256=checkpoint_sha256, artifacts=artifacts, videos=video_info)
    temporary = out / 'completion.json.partial'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(out / 'completion.json')
    return manifest


def verify_completion(out, checkpoint_sha256):
    """Parent-side standard-library proof; child exit code zero alone never suffices."""
    out = Path(out)
    summary = _summary(out, checkpoint_sha256)
    manifest = json.loads(_safe_file(out, 'completion.json').read_text())
    if (manifest.get('version') != 1 or manifest.get('steps') != 1000
            or manifest.get('policy_steps') != 999 or manifest.get('training_updates') != 0
            or manifest.get('checkpoint_sha256') != checkpoint_sha256):
        raise RuntimeError('Invalid replay completion manifest')
    expected = {'summary.json', 'trajectory.npz', 'joint_mapping.json', 'events.jsonl',
                'replay_config.yaml', 'encoder_preflight.json', *summary['videos']}
    if set(manifest.get('artifacts', {})) != expected or set(manifest.get('videos', {})) != set(summary['videos']):
        raise RuntimeError('Replay completion artifact set is incomplete')
    for name, proof in manifest['artifacts'].items():
        path = out / name if name == 'events.jsonl' else _safe_file(out, name)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Missing completion artifact')
        if path.stat().st_size != proof['bytes'] or file_digest(path) != proof['sha256']:
            raise RuntimeError('Replay artifact changed after validation: ' + name)
    for proof in manifest['videos'].values():
        if proof['frames'] < 1 or proof['width'] != 640 or proof['height'] != 360:
            raise RuntimeError('Invalid completed video metadata')
    return summary


def verify_upload_receipt(out):
    parent = Path(out).parent
    receipt = json.loads(_safe_file(parent, 'replay_bundle_receipt.json').read_text())
    bundle = _safe_file(parent, 'replay_bundle.zip')
    if (receipt.get('status') != 'uploaded' or receipt.get('bytes') != bundle.stat().st_size
            or receipt.get('sha256') != file_digest(bundle)):
        raise RuntimeError('Replay bundle upload has no matching successful receipt')
    return receipt