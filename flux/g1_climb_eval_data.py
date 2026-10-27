"""Evaluation-only data transport; training downloaders remain unchanged."""
from pathlib import Path
from flux.g1_climb_contract import input_source
text=input_source(Path(__file__).parent)
old='from flux.verified_download import download'
if text.count(old)!=1:raise ValueError('Climb input downloader anchor changed')
exec(compile(text.replace(old,'from flux.parallel_download import download'),__file__,'exec'),globals())