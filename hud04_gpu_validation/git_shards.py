"""Local Git shard preparation/reassembly. No network, credentials or GPU calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path

CHUNK = 48 * 1024**2
INPUTS = {
    'base': (123410856, '490d7d1b8b927f42303816c1a40f440c05ef5393aec950724ab897fad45c643f'),
    'data': (470500407, 'f2e3c5de404b68b9ac1e6849e9d4f02aa5fcf88bc378cc45dc2b8865ac3aea79'),
}


def safe(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink rejected')
    return path


def digest(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def build(sources, destination):
    dest = safe(destination)
    # Verify ALL original inputs before creating a publication candidate.
    for name, (size, sha) in INPUTS.items():
        p = safe(sources[name])
        if not p.is_file() or p.stat().st_size != size or digest(p) != sha:
            raise ValueError('Original input hash/size mismatch: ' + name)
    dest.mkdir(exist_ok=False, parents=True)
    manifest = {'schema': 1, 'chunk_bytes': CHUNK, 'inputs': {}}
    for name, (size, sha) in INPUTS.items():
        parts = []
        with safe(sources[name]).open('rb') as src:
            for index in range(math.ceil(size / CHUNK)):
                block = src.read(CHUNK)
                filename = f'{name}.zip.part-{index:04d}'
                with (dest / filename).open('xb') as out:
                    out.write(block)
                parts.append({'path': filename, 'bytes': len(block),
                              'sha256': hashlib.sha256(block).hexdigest()})
            if src.read(1):
                raise ValueError('Input changed during build')
        manifest['inputs'][name] = {'bytes': size, 'sha256': sha, 'parts': parts}
    with (dest / 'manifest.json').open('x') as out:
        json.dump(manifest, out, indent=2)
    return manifest


def restore(directory, destination):
    root = safe(directory)
    mf = safe(root / 'manifest.json')
    if mf.stat().st_size > 65536:
        raise ValueError('Manifest too large')
    manifest = json.loads(mf.read_text())
    if (set(manifest) != {'schema', 'chunk_bytes', 'inputs'}
            or manifest['schema'] != 1 or manifest['chunk_bytes'] != CHUNK
            or set(manifest['inputs']) != set(INPUTS)):
        raise ValueError('Manifest schema mismatch')
    expected_paths = {'manifest.json'}
    for name, (size, sha) in INPUTS.items():
        item = manifest['inputs'][name]
        if (set(item) != {'bytes', 'sha256', 'parts'} or item['bytes'] != size
                or item['sha256'] != sha or len(item['parts']) != math.ceil(size / CHUNK)):
            raise ValueError('Trusted input binding mismatch')
        for index, part in enumerate(item['parts']):
            filename = f'{name}.zip.part-{index:04d}'
            if (set(part) != {'path', 'bytes', 'sha256'} or part['path'] != filename
                    or part['bytes'] != min(CHUNK, size - index * CHUNK)):
                raise ValueError('Part ordering/name/size mismatch')
            p = safe(root / filename)
            expected_paths.add(filename)
            if (not p.is_file() or p.stat().st_size != part['bytes']
                    or digest(p) != part['sha256']):
                raise ValueError('Missing/corrupt/LFS part: ' + filename)
    if {p.name for p in root.iterdir()} != expected_paths:
        raise ValueError('Unexpected shard directory entries')
    dest = safe(destination)
    dest.mkdir(exist_ok=False, parents=True)
    results = {}
    for name, (size, sha) in INPUTS.items():
        p = dest / (name + '.zip')
        h = hashlib.sha256()
        with p.open('xb') as out:
            for part in manifest['inputs'][name]['parts']:
                with safe(root / part['path']).open('rb') as src:
                    for block in iter(lambda: src.read(1024**2), b''):
                        h.update(block)
                        out.write(block)
        if p.stat().st_size != size or h.hexdigest() != sha:
            raise ValueError('Full input hash mismatch; output must not be used')
        results[name] = {'bytes': size, 'sha256': sha, 'hash_verified': True}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    b = sub.add_parser('build')
    b.add_argument('--base', required=True)
    b.add_argument('--data', required=True)
    b.add_argument('--destination', required=True)
    r = sub.add_parser('restore')
    r.add_argument('--directory', required=True)
    r.add_argument('--destination', required=True)
    a = p.parse_args()
    result = (build({'base': a.base, 'data': a.data}, a.destination)
              if a.operation == 'build' else restore(a.directory, a.destination))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()