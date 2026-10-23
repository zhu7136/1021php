"""Merge ordinary Git blobs locally. No network, LFS, credentials or fallback."""
import hashlib
import json
from pathlib import Path
import stat

PART_BYTES = 48 * 1024 * 1024
PART_COUNT = 5
MANIFEST_NAME = 'manifest.json'


def assemble_parts(source: Path, destination: Path, archive_bytes: int,
                   archive_sha256: str) -> None:
    """Pin the assembled payload to the bootstrap's immutable archive hash.

    The manifest helps diagnose individual corruption; it cannot approve a new
    archive. Preserve partial output on failure, and never overwrite a file.
    """
    source, destination = Path(source), Path(destination)
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise ValueError('Parts directory must be an existing absolute directory')
    manifest = source / MANIFEST_NAME
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 16384:
        raise ValueError('Missing, unsafe or oversized parts manifest')
    record = json.loads(manifest.read_text())
    if (record.get('format') != 'hud04-v8-git-parts-1'
            or record.get('archive_bytes') != archive_bytes
            or record.get('archive_sha256') != archive_sha256
            or record.get('part_bytes') != PART_BYTES):
        raise ValueError('Parts manifest does not describe the approved archive')
    parts = record.get('parts')
    if not isinstance(parts, list) or len(parts) != PART_COUNT:
        raise ValueError('Expected exactly five ordered Git parts')
    total = 0
    for index, part in enumerate(parts):
        expected = min(PART_BYTES, archive_bytes-total)
        if (not isinstance(part, dict) or part.get('name') != f'archive.part{index:03d}'
                or type(part.get('bytes')) is not int or part['bytes'] != expected or expected <= 0
                or not isinstance(part.get('sha256'), str)
                or len(part['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in part['sha256'])):
            raise ValueError('Invalid part name, order, size or hash')
        path = source / part['name']
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('Missing or unsafe Git part')
        if path.stat().st_size != expected:
            raise ValueError('Git part size mismatch; LFS pointers are not supported')
        if path.resolve() == destination.resolve():
            raise ValueError('Source and destination must differ')
        total += expected
    if total != archive_bytes:
        raise ValueError('Git parts total size mismatch')
    count = 0
    combined = hashlib.sha256()
    with destination.open('xb') as target:
        for part in parts:
            digest = hashlib.sha256()
            part_count = 0
            with (source / part['name']).open('rb') as reader:
                while chunk := reader.read(1024 * 1024):
                    part_count += len(chunk)
                    if part_count > part['bytes']:
                        raise ValueError('Git part exceeds expected size')
                    digest.update(chunk)
                    combined.update(chunk)
                    target.write(chunk)
            if part_count != part['bytes'] or digest.hexdigest() != part['sha256']:
                raise ValueError('Git part SHA256 mismatch; refusing to train')
            count += part_count
    if count != archive_bytes or combined.hexdigest() != archive_sha256:
        raise ValueError('Assembled archive SHA256 mismatch; refusing to train')