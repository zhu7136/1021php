"""Pinned, independently retrieved task048 checkpoint; no network at import."""
from pathlib import Path
from flux.formal_contract import replace_once
from flux import formal_input
_source = Path(formal_input.__file__).read_text()
for _old, _new in [
    ('ARCHIVE_BYTES = 30751697', 'ARCHIVE_BYTES = 32585651'),
    (formal_input.ARCHIVE_SHA256, 'bea9ea4f56c4cb1efbb999656c9a28fbdc9f2ce233a5b2dd3c5fb4fe21aacbc0'),
    (formal_input.CHECKPOINT_SHA256, '265bb2eff169a08c890f63f70fd3b8782db23fed9ebcb44ebbfcb0fe42367718'),
    (formal_input.MEMBER, 'training/HuD04_WBT/20261025_060558-php-teacher-fullscale4096-1h-locomotion/model_smoke_final.pt'),
    ("SOURCE_TASK = 'TASK_20261025_033'", "SOURCE_TASK = 'TASK_20261025_048'"),
    ('SOURCE_ITERATION = 955', 'SOURCE_ITERATION = 3847'),
    ('MAX_UPDATES = 10000', 'MAX_UPDATES = 1000'),
]:
    _source = replace_once(_source, _old, _new)
exec(compile(_source, __file__, 'exec'), globals())