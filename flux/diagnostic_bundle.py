"""Bounded diagnostic packaging, with separate raw-data and transfer budgets."""
from pathlib import Path
import zipfile

MAX_RAW_BYTES=128*1024*1024
MAX_BUNDLE_BYTES=32*1024*1024

def build_bundle(directory,destination,max_raw=MAX_RAW_BYTES,max_bundle=MAX_BUNDLE_BYTES):
    directory,destination=Path(directory),Path(destination)
    assert directory.is_dir() and not directory.is_symlink()
    files=sorted(f for f in directory.glob('*') if f.suffix in ('.json','.jsonl','.yaml'))
    assert files and all(f.is_file() and not f.is_symlink() for f in files),'Unsafe diagnostic file'
    raw=sum(f.stat().st_size for f in files)
    assert raw<=max_raw,'Diagnostic raw-data budget exceeded'
    assert not destination.exists(),'Fresh bundle destination required'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for source in files:bundle.write(source,arcname=source.name)
    size=destination.stat().st_size
    assert size<=max_bundle,'Diagnostic compressed-transfer budget exceeded'
    with zipfile.ZipFile(destination) as bundle:
        assert bundle.testzip() is None,'Diagnostic ZIP integrity failure'
        assert sorted(bundle.namelist())==[f.name for f in files]
    return dict(raw_bytes=raw,compressed_bytes=size,raw_limit_bytes=max_raw,
                compressed_limit_bytes=max_bundle,files=[f.name for f in files])