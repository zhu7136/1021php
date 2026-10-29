"""Offline input preparation only; this module never launches a worker."""
from pathlib import Path
from local_shards import reassemble
from scoped_transport import deadline

NAMES = ['core.zip', 'train-motion.npz', 'eval-motion.npz']
KINDS = ['core', 'train_motion', 'eval_motion']


def materialize(directory, work, *, manifest_sha256, absolute_deadline, unpack_archive):
    work = Path(work)
    with deadline(absolute_deadline):
        receipt = reassemble(directory, work/'reconstructed-inputs', manifest_sha256=manifest_sha256)
        if [r['name'] for r in receipt['files']] != NAMES:
            raise ValueError('Expected exactly three ordered trial inputs')
        restored = work/'reconstructed-inputs'
        payload = work/'payload'
        unpack_archive(restored/'core.zip', payload)
        for name in NAMES[1:]:
            target = payload/'formal_data'/name
            if target.exists() or target.is_symlink():
                raise FileExistsError('Refuse to overwrite motion input')
            target.parent.mkdir(parents=True, exist_ok=True)
            (restored/name).rename(target)
    return [{'kind': kind, **{k: row[k] for k in ('bytes', 'sha256')}}
            for kind, row in zip(KINDS, receipt['files'])]