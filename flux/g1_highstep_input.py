"""Same strict extraction/hash validation, pinned official high-speed manifest."""
from pathlib import Path
from flux.g1_highstep_contract import adapt
from flux.formal_contract import replace_once
text=adapt(Path(__file__).with_name('g1_lowstep_input.py').read_text())
old="""    h=hashlib.sha256();remaining=control['bytes']
    with urllib.request.urlopen(control['url'],timeout=45) as response,archive.open('xb') as out:
        if urllib.parse.urlsplit(response.url).hostname!='storage.googleapis.com':raise ValueError('Unexpected input redirect')
        while remaining:
            block=response.read(min(1048576,remaining))
            if not block:raise ValueError('Truncated high-step input')
            h.update(block);out.write(block);remaining-=len(block)
        if response.read(1):raise ValueError('Oversized high-step input')
    if h.hexdigest()!=control['sha256']:raise ValueError('High-step archive hash mismatch')"""
new="""    from flux.verified_download import download
    download(control['url'],archive,control['bytes'],control['sha256'],
             destination.parent/'data-download.json')"""
text=replace_once(text,old,new)
exec(compile(text,__file__,'exec'),globals())