"""Fail-closed delivery of task276's independently verified final checkpoint."""
import hashlib,urllib.parse,urllib.request
from pathlib import Path
SOURCE_TASK='TASK_20261026_276'
SOURCE_RUN='wj7an49u'
SOURCE_ITERATION=1718
SOURCE_UPDATES=1720
CHECKPOINT_BYTES=10346235
CHECKPOINT_SHA256='c41e7ea49b13b1efcc664ff6e5d834a26b8e6408f5a7cb711ea2927cf772eb81'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def validate_control(control):
    if set(control)!={'url','sha256','bytes','source_task','source_run'}:
        raise ValueError('Invalid276 checkpoint control fields')
    if (type(control['bytes']) is not int or control['bytes']!=CHECKPOINT_BYTES or
        control['sha256']!=CHECKPOINT_SHA256 or control['source_task']!=SOURCE_TASK or control['source_run']!=SOURCE_RUN):
        raise ValueError('Not the verified276 final checkpoint')
    parsed=urllib.parse.urlsplit(control['url'])
    if parsed.scheme!='https' or parsed.hostname!='storage.googleapis.com' or parsed.username or parsed.password:
        raise ValueError('Only scoped W&B storage delivery accepted')
    return control

def fetch_checkpoint(control,destination):
    validate_control(control)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    path=destination/'source276_final.pt';remaining=CHECKPOINT_BYTES
    with urllib.request.urlopen(control['url'],timeout=45) as response,path.open('xb') as out:
        if urllib.parse.urlsplit(response.url).hostname!='storage.googleapis.com':raise ValueError('Unexpected input redirect')
        while remaining:
            block=response.read(min(1048576,remaining))
            if not block:raise ValueError('Truncated276 checkpoint')
            out.write(block);remaining-=len(block)
        if response.read(1):raise ValueError('Oversized276 checkpoint')
    if digest(path)!=CHECKPOINT_SHA256:raise ValueError('276 checkpoint hash mismatch')
    return path