"""Offline, hash-pinned Git input assembly. No network or training side effects."""
import hashlib
import json
from pathlib import Path
import re

from flux.prepare_inputs import regular, verify_parts, verify_file


def assemble_git_input(repo_root, manifest_name, manifest_sha256, output, expected_bytes, expected_sha256):
    """One ordinary Git file or multiple parts; never follow a network fallback.

    The manifest hash is pinned by the control object (and source commit).
    This function does not confer publication or GPU authorization.
    """
    root=Path(repo_root).absolute()
    output=Path(output)
    for value in (manifest_sha256,expected_sha256):
        if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
            raise ValueError('Invalid SHA256 pin')
    if type(expected_bytes) is not int or not 0<expected_bytes<400*1024*1024:
        raise ValueError('Invalid archive size pin')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refusing output overwrite')
    for parent in output.parents:
        if parent.is_symlink():raise ValueError('Symlink output parent')
    manifest_path=regular(root,manifest_name)
    raw=manifest_path.read_bytes() if manifest_path.stat().st_size<=1024*1024 else b''
    if not raw or hashlib.sha256(raw).hexdigest()!=manifest_sha256:
        raise ValueError('Manifest hash mismatch or oversized manifest')
    manifest=json.loads(raw)
    if manifest.get('delivery')!='ordinary-git' or len(manifest.get('archives',[]))!=1:
        raise ValueError('Expected exactly one ordinary Git input archive')
    archive=manifest['archives'][0]
    if archive.get('name')!='input.zip' or archive.get('bytes')!=expected_bytes or archive.get('sha256')!=expected_sha256:
        raise ValueError('Archive disagrees with control pins')
    parts=archive.get('parts',[])
    if not 1<=len(parts)<=64 or any(type(p.get('bytes')) is not int or p['bytes']<=0 for p in parts):
        raise ValueError('Invalid part count or size')
    verify_parts(manifest_path.parent,manifest)
    # All part hashes pass before any output is created. Exclusive output,
    # bounded copies and final hash also detect changes between verify/copy.
    received=0
    with output.open('xb') as dest:
        for part in parts:
            with regular(manifest_path.parent,part['path']).open('rb') as source:
                remaining=part['bytes']
                while remaining:
                    block=source.read(min(1024*1024,remaining))
                    if not block:raise ValueError('Part truncated during assembly')
                    dest.write(block);remaining-=len(block);received+=len(block)
                if source.read(1):raise ValueError('Part grew during assembly')
    verify_file(output,archive)
    return {'delivery':'ordinary-git','parts':len(parts),'bytes_received':received,'sha256_verified':True}