"""Stdlib-only delivery of the immutable, already-verified task027 checkpoint."""
import hashlib
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

ARCHIVE_BYTES = 113246576
ARCHIVE_SHA256 = 'de8c843653b9a861b0a932f5e7581f16e6fc700488786c56b3c138ef0b314d44'
CHECKPOINT_BYTES = 10345851
CHECKPOINT_SHA256 = 'fcb1fa9bf1782692c17f24ae9ede3983614c989be9132575be34b450fd326dac'
MEMBER = 'training/HuD04_WBT/20261025_023808-php-locomotion-teacher-smoke-locomotion/model_smoke_final.pt'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_control(control):
    if set(control) != {'url', 'source_task', 'checkpoint_sha256'}:
        raise ValueError('Unexpected resume input fields')
    url = urllib.parse.urlsplit(control['url'])
    if (url.scheme != 'https' or url.hostname != 'storage.googleapis.com'
            or url.username or url.password or url.port not in (None, 443) or not url.query):
        raise ValueError('Only a scoped signed GCS read is permitted')
    if control['source_task'] != 'TASK_20261025_027' or control['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Unverified checkpoint source')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Resume input redirects are not allowed')


def extract_checkpoint(archive, destination):
    if Path(archive).stat().st_size != ARCHIVE_BYTES or digest(archive) != ARCHIVE_SHA256:
        raise ValueError('Resume archive size/hash mismatch')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist().count(MEMBER) != 1 or bundle.getinfo(MEMBER).file_size != CHECKPOINT_BYTES:
            raise ValueError('Unexpected checkpoint member')
        raw = bundle.read(MEMBER)
    if len(raw) != CHECKPOINT_BYTES or hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('Resume checkpoint size/hash mismatch')
    with Path(destination).open('xb') as stream:
        stream.write(raw)
    Path(destination).chmod(0o444)
    return Path(destination)


def fetch_checkpoint(control, directory):
    validate_control(control)
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    archive = directory / 'source.zip'
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(control['url'], timeout=45) as incoming, archive.open('xb') as output:
        remaining = ARCHIVE_BYTES
        while remaining:
            block = incoming.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError('Incomplete resume archive')
            output.write(block)
            remaining -= len(block)
        if incoming.read(1):
            raise ValueError('Oversized resume archive')
    return extract_checkpoint(archive, directory / 'source_checkpoint.pt')