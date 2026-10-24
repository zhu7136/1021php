"""Exactly one read-only replay, with a 900s whole-worker deadline and scoped output upload.

gm-run wraps this bootstrap only. Never invoke the training entrypoint or save a model.
No credentials are embedded in source. An optional signed upload grants one output-file PUT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKER_LIMIT_SECONDS = 900
OUT = ROOT / "outputs/flux_replay_latest_1000"


def event(name, **fields):
    print(json.dumps(dict(event=name, **fields), sort_keys=True), flush=True)


def upload_bundle(url: str, headers: dict):
    if not OUT.exists():
        return False
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or not (parsed.hostname == "storage.googleapis.com"
                    or str(parsed.hostname).endswith(".storage.googleapis.com"))):
        raise ValueError("Only a scoped signed Google Cloud Storage PUT is accepted")
    for name in headers:
        if name.lower() != "content-type" and not name.lower().startswith("x-goog-"):
            raise ValueError("Unexpected upload header; never forward general API credentials")
    bundle = ROOT / "outputs/replay_bundle.zip"
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    if len(files) > 2048 or sum(p.stat().st_size for p in files) > 80 * 1024 * 1024:
        raise ValueError("Replay output exceeds its upload bound")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in files:
            if path.is_symlink():
                raise ValueError("Output symlinks are forbidden")
            target.write(path, arcname=path.relative_to(OUT).as_posix())
    payload = bundle.read_bytes()
    request = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
    # No redirects or retry: the scoped upload capability must only reach its own service.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    with urllib.request.build_opener(NoRedirect).open(request, timeout=45) as response:
        if response.status not in (200, 201, 204):
            raise RuntimeError("Diagnostic upload was not accepted")
    event("replay_bundle_uploaded", bytes=len(payload), files=len(files),
          sha256=hashlib.sha256(payload).hexdigest(), filename="replay_bundle.zip")
    return True


def worker():
    from source_preflight import verify_sources
    from runtime_preflight import verify_runtime
    event("replay_source_verified", **verify_sources(ROOT))
    event("replay_runtime_verified", **verify_runtime(ROOT))
    from git_parts_v8 import assemble_parts
    from git_smoke_v8 import ARCHIVE_BYTES, ARCHIVE_SHA256, extract_candidates
    work = ROOT / "outputs/replay_input_data"
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "v8.zip"
    assemble_parts(ROOT / "flux/data/hud04-v8-parts", archive, ARCHIVE_BYTES, ARCHIVE_SHA256)
    dataset = work / "kinematic-candidates-56"
    extract_candidates(archive, dataset)
    event("replay_v8_candidates_verified", motion_files=56)
    from offline_dependencies import install
    install(ROOT)
    from train_hud04 import prepare_runtime
    runtime = prepare_runtime()
    sys.path[:0] = [str(ROOT / ".flux-deps"), str(ROOT), str(runtime)]
    os.chdir(ROOT)
    import numpy as np
    from wbt_training.utils.registry import validate_bundle
    from holosoma.config_values.robot import hu_d04
    validate_bundle(dataset)
    for path in sorted(dataset.glob("*_motion.npz")):
        with np.load(path, allow_pickle=False) as data:
            assert data["joint_names"].tolist() == hu_d04.dof_names
            assert data["body_names"].tolist() == hu_d04.body_names
    from replay_teacher_finite import main
    main(dataset)
    event("replay_worker_complete", control_steps=1000, training_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-child", action="store_true")
    parser.add_argument("--upload-url", required=True)
    parser.add_argument("--upload-headers", default="{}")
    args = parser.parse_args()
    if args.runtime_child:
        try:
            worker()
        finally:
            try:
                upload_bundle(args.upload_url, json.loads(args.upload_headers))
            except Exception as exc:
                # Exception text may contain a signed URL; emit only its type.
                event("replay_bundle_upload_failed", exception_type=type(exc).__name__)
        return
    from runtime_preflight import resolve_isaac_launcher
    launcher = resolve_isaac_launcher()
    child_env = os.environ.copy()
    child_env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0", WANDB_MODE="disabled")
    # The compute worker never receives the controller's API keys.
    for name in ("WANDB_API_KEY", "OPENAI_API_KEY", "github", "key", "FLUX_API_KEY"):
        child_env.pop(name, None)
    event("replay_worker_start", worker_limit_seconds=WORKER_LIMIT_SECONDS,
          gpu="1x4090D24GB", num_envs=1, max_control_steps=1000,
          automatic_retries=0, training_updates=0)
    child = subprocess.Popen([str(launcher), str(Path(__file__).resolve()),
        "--runtime-child", "--upload-url", args.upload_url, "--upload-headers", args.upload_headers],
        env=child_env, start_new_session=True)
    try:
        from startup_diagnostics import wait_for_child
        result = wait_for_child(child, ROOT, WORKER_LIMIT_SECONDS)
    except subprocess.TimeoutExpired:
        event("replay_worker_deadline_reached", seconds=WORKER_LIMIT_SECONDS)
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise SystemExit(124)
    if result != 0:
        raise SystemExit(result)
    event("replay_entrypoint_complete", control_steps=1000, training_updates=0)


if __name__ == "__main__":
    main()