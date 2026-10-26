"""Read-only delivery of the verified task283 checkpoint; old lineage unchanged."""
from pathlib import Path
from flux import g1_resume276_input as previous

text=Path(previous.__file__).read_text()
for old,new in [('276','283'),('wj7an49u','9fe07g0l'),('SOURCE_ITERATION=1718','SOURCE_ITERATION=2594'),
                ('SOURCE_UPDATES=1720','SOURCE_UPDATES=2597'),
                ('c41e7ea49b13b1efcc664ff6e5d834a26b8e6408f5a7cb711ea2927cf772eb81',
                 '416f68e33c18bb39a354f8de66277cb77967b3ac9bbb6022ee631262442174f2')]:
    text=text.replace(old,new)
exec(compile(text,'verified_eval283_input','exec'),globals())