"""Pinned, independently verified high-speed292 final checkpoint only."""
from pathlib import Path
source=Path(__file__).with_name('g1_resume224_input.py').read_text()
for old,new in [('224','292'),('r1f8gw5h','00vxykeq'),('SOURCE_ITERATION=841','SOURCE_ITERATION=833'),
                ('SOURCE_UPDATES=842','SOURCE_UPDATES=834'),
                ('aca12c5bbda04c7cfa4ff8f85cb10b414939dce4990c2f015ccdbc43732a982d','c254b82205e139ade4698a41f309849569712c069a4404d7003e53e08e1f756a')]:
    assert old in source
    source=source.replace(old,new)
exec(compile(source,__file__,'exec'),globals())