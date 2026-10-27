"""r20 adapter: identical bounded native bootstrap, observable input transfer."""
import json
from pathlib import Path
import sys
import zipfile
import bootstrap_base as base
from download_progress import monitored_download
from scoped_transport import digest

_raw_transfer=base.transfer
_download_state={}


def transfer(cap,path,**kwargs):
    path=Path(path)
    if kwargs['method']=='GET':
        try:
            return monitored_download(_raw_transfer,cap,path,state=_download_state,
                emit=lambda data:print(json.dumps(data),flush=True),**kwargs)
        finally:
            if _download_state:
                (path.parent/'download-status.json').write_text(json.dumps(_download_state,indent=2)+'\n')
    if kwargs['method']=='PUT' and path.name=='flux-result.zip':
        if _download_state:
            with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as archive:
                if 'download-status.json' in archive.namelist():raise ValueError('Duplicate progress result')
                archive.writestr('download-status.json',json.dumps(_download_state,indent=2)+'\n')
            kwargs=dict(kwargs,size=path.stat().st_size,sha256=digest(path))
    return _raw_transfer(cap,path,**kwargs)


def main():
    base.transfer=transfer
    return base.main()


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        print(json.dumps(dict(event='bootstrap_failure',error_type=type(error).__name__,passed=False)),flush=True)
        sys.exit(1)