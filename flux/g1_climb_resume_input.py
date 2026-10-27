"""Pinned, independently CPU/hash-verified first climb finals only."""
import hashlib,urllib.parse
from pathlib import Path
from flux.g1_climb_contract import kind
SOURCES={
 'low':('TASK_20261027_009','fz2e14cf',659,660,'1a5f674cbea1afbb045df40b692535b892e2e116f3bd162fc9654c8effb8c2cb'),
 'high':('TASK_20261027_008','6gggzoyy',656,657,'ce57400fba659eec2d92ac6c2ae0ea36b3dd601a223a8adf02cfa71064344255')}
SOURCE_TASK,SOURCE_RUN,SOURCE_ITERATION,SOURCE_UPDATES,CHECKPOINT_SHA256=SOURCES[kind()]
CHECKPOINT_BYTES=10346107
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_control(control):
    if set(control)!={'url','sha256','bytes','source_task','source_run'}:raise ValueError('Invalid climb source fields')
    if type(control['bytes']) is not int or control['bytes']!=CHECKPOINT_BYTES or control['sha256']!=CHECKPOINT_SHA256 or control['source_task']!=SOURCE_TASK or control['source_run']!=SOURCE_RUN:raise ValueError('Wrong climb source')
    p=urllib.parse.urlsplit(control['url'])
    if p.scheme!='https' or p.hostname!='storage.googleapis.com' or p.username or p.password:raise ValueError('Scoped storage URL required')
    return control
def fetch_checkpoint(control,destination):
    from flux.verified_download import download
    validate_control(control);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    return download(control['url'],destination/'source_climb_final.pt',CHECKPOINT_BYTES,CHECKPOINT_SHA256,destination.parent/'checkpoint-download.json')