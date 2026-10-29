"""Offline pinned binary shards. No network, credentials, or training calls."""
import hashlib
import json
from pathlib import Path
import re

MAX_CHUNK = 48 * 1024**2
MANIFEST = 'shard-manifest.json'
RESERVED = {MANIFEST, 'reassembly-receipt.json'}
HEX = re.compile(r'[0-9a-f]{64}\Z')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}\Z')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def integer(n):
    return type(n) is int and n > 0


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)


def inspect_manifest(directory, manifest_sha256):
    directory = Path(directory)
    path = directory / MANIFEST
    if directory.is_symlink() or not directory.is_dir() or path.is_symlink():
        raise ValueError('Expected ordinary shard directory')
    if not isinstance(manifest_sha256, str) or not HEX.fullmatch(manifest_sha256):
        raise ValueError('Expected independent manifest SHA256 pin')
    if path.stat().st_size > 1024**2 or sha256(path) != manifest_sha256:
        raise ValueError('Manifest integrity')
    manifest = json.loads(path.read_text())
    if set(manifest) != {'version', 'chunk_bytes', 'files'} or type(manifest['version']) is not int or manifest['version'] != 1:
        raise ValueError('Manifest format')
    cap = manifest['chunk_bytes']
    if not integer(cap) or cap > MAX_CHUNK:
        raise ValueError('Chunk bound')
    files = manifest['files']
    if not isinstance(files, list) or not 1 <= len(files) <= 16:
        raise ValueError('File count')
    expected = {MANIFEST}
    names = set()
    for i, row in enumerate(files):
        if set(row) != {'name', 'bytes', 'sha256', 'shards'}:
            raise ValueError('File fields')
        name = row['name']
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names or name in RESERVED:
            raise ValueError('Unsafe or duplicate output name')
        names.add(name)
        if not integer(row['bytes']) or not isinstance(row['sha256'], str) or not HEX.fullmatch(row['sha256']):
            raise ValueError('File size/hash')
        shards = row['shards']
        count = (row['bytes']+cap-1)//cap
        if not isinstance(shards, list) or len(shards) != count or count > 10000:
            raise ValueError('Shard count')
        for j, shard in enumerate(shards):
            if set(shard) != {'path', 'bytes', 'sha256'}:
                raise ValueError('Shard fields')
            expected_path = f'f{i:02d}-p{j:05d}.bin'
            size = cap if j+1 < count else row['bytes']-cap*j
            if shard['path'] != expected_path or type(shard['bytes']) is not int or shard['bytes'] != size:
                raise ValueError('Shard name/order/size')
            if not isinstance(shard['sha256'], str) or not HEX.fullmatch(shard['sha256']):
                raise ValueError('Shard hash')
            expected.add(expected_path)
    actual = {p.name for p in directory.iterdir()}
    if actual != expected:
        raise ValueError('Missing or unlisted shard')
    for name in expected:
        item = directory / name
        if item.is_symlink() or not item.is_file():
            raise ValueError('Only regular input files permitted')
    return manifest


def split_files(inputs, directory, *, chunk_bytes=MAX_CHUNK):
    if not integer(chunk_bytes) or chunk_bytes > MAX_CHUNK:
        raise ValueError('Chunk bound')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = []
    for i, row in enumerate(inputs):
        name = row['name']
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in RESERVED:
            raise ValueError('Unsafe input name')
        if name in [f['name'] for f in files]:
            raise ValueError('Duplicate input name')
        source = Path(row['source'])
        if source.is_symlink() or not source.is_file() or source.stat().st_size != row['bytes'] or not integer(row['bytes']):
            raise ValueError('Source size/type')
        chunks = []
        whole = hashlib.sha256()
        size = 0
        with source.open('rb') as stream:
            j = 0
            while True:
                block = stream.read(chunk_bytes)
                if not block: break
                path = f'f{i:02d}-p{j:05d}.bin'
                with (directory/path).open('xb') as target:
                    target.write(block)
                chunks.append({'path': path, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
                whole.update(block)
                size += len(block)
                j += 1
        if size != row['bytes'] or whole.hexdigest() != row['sha256']:
            raise ValueError('Pinned source integrity')
        files.append({'name': name, 'bytes': size, 'sha256': whole.hexdigest(), 'shards': chunks})
    manifest = {'version': 1, 'chunk_bytes': chunk_bytes, 'files': files}
    write_json_new(directory/MANIFEST, manifest)
    pin = sha256(directory/MANIFEST)
    inspect_manifest(directory, pin)
    return {'manifest_sha256': pin, 'total_bytes': sum(f['bytes'] for f in files),
            'files': len(files), 'shards': sum(len(f['shards']) for f in files)}


def reassemble(directory, destination, *, manifest_sha256):
    directory = Path(directory)
    manifest = inspect_manifest(directory, manifest_sha256)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for row in manifest['files']:
        whole = hashlib.sha256()
        total = 0
        with (destination/row['name']).open('xb') as target:
            for shard in row['shards']:
                path = directory/shard['path']
                if path.stat().st_size != shard['bytes']:
                    raise ValueError('Shard size')
                part = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024**2), b''):
                        target.write(block)
                        part.update(block)
                        whole.update(block)
                        total += len(block)
                if part.hexdigest() != shard['sha256']:
                    raise ValueError('Shard integrity')
        if total != row['bytes'] or whole.hexdigest() != row['sha256']:
            raise ValueError('Whole file integrity')
    receipt = {'manifest_sha256': manifest_sha256,
               'files': [{k: row[k] for k in ('name', 'bytes', 'sha256')} for row in manifest['files']]}
    write_json_new(destination/'reassembly-receipt.json', receipt)
    return receipt