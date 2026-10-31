"""Native save/reload inside the log directory, then verified bounded export.

No simulator imports and no relaxation of the native checkpoint path guard.
"""
import copy
import hashlib
from pathlib import Path

NAME='student_dagger_ppo_validation.pt'
MAX_BYTES=150*1024**2

def _regular_inside(path,root):
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('Checkpoint must be a regular file within its declared root')

def _digest(path):
    h=hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda:src.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def finalize_native_checkpoint(algo,output,cpu_state,require_equal):
    output=Path(output)
    if output.is_symlink():raise ValueError('Linked output directory rejected')
    output=output.resolve(strict=True)
    raw=Path(algo.logging_helper.log_dir)
    if raw.is_symlink() or any(p.is_symlink() for p in raw.parents):
        raise ValueError('Linked native log directory rejected')
    logdir=raw.resolve(strict=True)
    if not logdir.is_dir() or logdir==output or not logdir.is_relative_to(output):
        raise ValueError('Expected nested native log directory inside bounded output')
    if Path(algo.log_dir).resolve(strict=True)!=logdir:
        raise ValueError('Algorithm and logging helper directories disagree')
    native=logdir/NAME;final=output/NAME;pending=output/(NAME+'.pending')
    if any(p.exists() or p.is_symlink() for p in (native,final,pending)):
        raise FileExistsError('Refuse to overwrite any existing checkpoint/export')
    model=cpu_state(algo.policy)
    optimizer=copy.deepcopy(algo.optimizer.state_dict())
    scalars={k:copy.deepcopy(getattr(algo,k)) for k in ('current_learning_iteration','learning_rate','ppo_coef')}
    readiness=list(algo.policy.loaded_teachers)
    # Use exactly the supported native save/load, preserving full-state metadata.
    algo.save(path=str(native))
    _regular_inside(native,logdir)
    size=native.stat().st_size
    if not 0<size<=MAX_BYTES:raise ValueError('Checkpoint export size bound exceeded')
    original_sha=_digest(native)
    algo.load(str(native))
    require_equal(model,cpu_state(algo.policy))
    require_equal(optimizer,algo.optimizer.state_dict())
    require_equal(scalars,{k:getattr(algo,k) for k in scalars})
    require_equal(readiness,list(algo.policy.loaded_teachers))
    if _digest(native)!=original_sha:raise ValueError('Native checkpoint changed during reload')
    written=0;h=hashlib.sha256()
    with native.open('rb') as src,pending.open('xb') as dst:
        for block in iter(lambda:src.read(1024**2),b''):
            written+=len(block)
            if written>MAX_BYTES:raise ValueError('Checkpoint grew during export')
            dst.write(block);h.update(block)
    if written!=size or h.hexdigest()!=original_sha or _digest(pending)!=original_sha:
        raise ValueError('Checkpoint export byte verification failed')
    if final.exists() or final.is_symlink():raise FileExistsError('Export destination appeared')
    pending.rename(final)
    _regular_inside(final,output)
    if _digest(final)!=original_sha:raise ValueError('Final export readback failed')
    return {'native_checkpoint':native.relative_to(output).as_posix(),
        'export_checkpoint':NAME,'checkpoint_bytes':size,'checkpoint_sha256':original_sha,
        'native_reload_verified':True,'model_optimizer_scalars_readiness_equal':True,
        'export_bytes_readback_verified':True,'native_path_guard_unchanged':True}