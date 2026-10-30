"""Explicitly resolved native depth wrapper; never imported by default.

Only a future authorized native diagnostic should resolve this module. The
CPU-only suite tests the mixin against source-extracted native tensor methods.
"""
import hashlib
import inspect
from pathlib import Path

from holosoma.managers.observation.terms.depth import WarpDepthImageObsTerm
from hud04_depth_diagnostics import NumericDepthEventsMixin

PINNED_DEPTH_SHA256 = '6d1483e1381e8ccb66aa6f23c236da2678783023a9b0f9d71af25942f133502c'
source = inspect.getsourcefile(WarpDepthImageObsTerm)
if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != PINNED_DEPTH_SHA256:
    raise ValueError('Native depth source does not match reviewed version')


class HuD04DiagnosticDepth(NumericDepthEventsMixin, WarpDepthImageObsTerm):
    pass