"""Immutable verified task033 source for one bounded continuation."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile
from flux.resume_input import NoRedirect, digest, validate_control as original_validate

ARCHIVE_BYTES = 30751697
ARCHIVE_SHA256 = 'b5ff326aad43ae9c654d419e07f3e9af55d5ad88267e3513b5833f752e735dc7'
CHECKPOINT_BYTES = 10346235
CHECKPOINT_SHA256 = '6b1bab179456d475af5c6cd223c36fc5e5c45f4abb13a7db31e7865ae2082bc9'
MEMBER = 'training/HuD04_WBT/20261025_034855-php-locomotion-teacher-formal-30m-locomotion/model_smoke_final.pt'
SOURCE_TASK = 'TASK_20261025_033'
SOURCE_ITERATION = 955
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