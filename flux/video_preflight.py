"""Check the real H.264 encoder before starting expensive simulation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def encoder_preflight(output, env):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(env)
    bindir = output / 'bin'
    bindir.mkdir()
    executable = shutil.which('ffmpeg', path=env.get('PATH', ''))
    if executable is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError('ffmpeg unavailable: install the pinned imageio-ffmpeg wheel before simulation') from exc
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if not Path(executable).is_file():
            raise RuntimeError('Bundled ffmpeg executable is missing')
        (bindir / 'ffmpeg').symlink_to(Path(executable).resolve())
        env['PATH'] = str(bindir.resolve()) + os.pathsep + env.get('PATH', '')
    executable = shutil.which('ffmpeg', path=env['PATH'])
    if executable is None:
        raise RuntimeError('ffmpeg not exposed on child PATH')
    version = subprocess.run([executable, '-version'], env=env, capture_output=True, text=True, timeout=10, check=True)
    probe = output / 'probe.bin'
    subprocess.run([executable, '-nostdin', '-v', 'error', '-n', '-f', 'lavfi', '-i',
        'color=c=black:s=16x16:r=1', '-frames:v', '1', '-c:v', 'libx264', '-pix_fmt',
        'yuv420p', '-f', 'mp4', str(probe)], env=env, capture_output=True, timeout=15, check=True)
    if probe.stat().st_size < 100:
        raise RuntimeError('H.264 encoder produced an empty probe')
    evidence = {'encoder_preflight_passed': True, 'version': version.stdout.splitlines()[0],
                'probe_bytes': probe.stat().st_size, 'codec': 'libx264', 'container': 'mp4'}
    (output / 'preflight.json').write_text(json.dumps(evidence, indent=2))
    return env, evidence


def fail_closed_entrypoint(function):
    """Do not let native simulator shutdown hang after a Python exception."""
    try:
        return function()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    _, evidence = encoder_preflight(args.output, os.environ)
    print(json.dumps({'event': 'eval283_encoder_preflight_passed', **evidence}), flush=True)