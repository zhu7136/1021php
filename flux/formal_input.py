"""Immutable task031 source for the single authorized 30-minute continuation."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile
from flux.resume_input import NoRedirect, digest, validate_control as original_validate

ARCHIVE_BYTES = 37909738
ARCHIVE_SHA256 = '548481a604a245ffee6b5011c4cda8b46dd655e10f691948bf20f1b748029de1'
CHECKPOINT_BYTES = 10346235
CHECKPOINT_SHA256 = 'b46bac428458245c8df035972fde8741072917b2b75f9ac3520f9c5f56d38356'
MEMBER = 'training/HuD04_WBT/20261025_032739-php-locomotion-teacher-smoke-locomotion/model_smoke_final.pt'
SOURCE_TASK = 'TASK_20261025_031'
SOURCE_ITERATION = 20
MAX_UPDATES = 10000


def validate_control(control):
    if (control.get('source_task') != SOURCE_TASK
            or control.get('checkpoint_sha256') != CHECKPOINT_SHA256):
        raise ValueError('Unverified formal continuation source')
    # Reuse the proven strict field and signed-GCS-URL checks without changing
    # task027's constants or monkey-patching its global state.
    from flux.resume_input import CHECKPOINT_SHA256 as old_sha
    original_validate(control | {'source_task': 'TASK_20261025_027', 'checkpoint_sha256': old_sha})


def extract_checkpoint(archive, destination):
    if Path(archive).stat().st_size != ARCHIVE_BYTES or digest(archive) != ARCHIVE_SHA256:
        raise ValueError('Formal source archive size/hash mismatch')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist().count(MEMBER) != 1 or bundle.getinfo(MEMBER).file_size != CHECKPOINT_BYTES:
            raise ValueError('Unexpected formal source member')
        raw = bundle.read(MEMBER)
    if len(raw) != CHECKPOINT_BYTES or hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('Formal source checkpoint size/hash mismatch')
    with Path(destination).open('xb') as stream:
        stream.write(raw)
    Path(destination).chmod(0o444)
    return Path(destination)


def fetch_checkpoint(control, directory):
    validate_control(control)
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(NoRedirect())
    archive = directory / 'source.zip'
    with opener.open(control['url'], timeout=45) as incoming, archive.open('xb') as output:
        remaining = ARCHIVE_BYTES
        while remaining:
            block = incoming.read(min(1024*1024, remaining))
            if not block:
                raise ValueError('Incomplete formal source archive')
            output.write(block)
            remaining -= len(block)
        if incoming.read(1):
            raise ValueError('Oversized formal source archive')
    return extract_checkpoint(archive, directory / 'source_checkpoint.pt')