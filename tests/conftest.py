import os

# CPU tests must also work on machines without a display or CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
