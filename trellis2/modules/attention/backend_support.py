# trellis2/modules/attention/backend_support.py
"""Which attention backends a card can run.

The app chooses its backend with this, and install.py decides with it which wheels to install.
install.py loads this file on its own, so it must not import anything else of the app.
"""

# Per backend, best first: the lowest CUDA compute capability its wheel has kernels for.
# flash_attn needs Ampere; the xformers wheel is built from Turing up, so a GTX 10xx gets neither.
MIN_CAPABILITY = {'flash_attn': (8, 0), 'xformers': (7, 5)}


def backends_for(capability) -> list:
    """Backends a card of this (major, minor) compute capability can run, best first."""
    return [name for name, lowest in MIN_CAPABILITY.items() if tuple(capability) >= lowest]
