"""Dormancy zoo: planted-behavior model organisms and the reawakening assay."""

import os

# Set before the first CUDA allocation (any zoo import comes first in every entry point).
# Readout and sampling tensors change size every step; without expandable segments the
# caching allocator fragments and reserves >8 GB on this 8 GB card (diagnosed 2026-11-10).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
