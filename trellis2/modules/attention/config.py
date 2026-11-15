# trellis2/modules/attention/config.py
from typing import *
import functools
import importlib
from .backend_support import MIN_CAPABILITY, backends_for

BACKEND = None  # resolved in __from_env
DEBUG = False

# The module each backend really needs. xformers.ops refuses to load beside a flash_attn it does not know.
_BACKEND_MODULES = {'flash_attn': 'flash_attn', 'xformers': 'xformers.ops'}


@functools.lru_cache(maxsize=None)
def detect_best_backend() -> str:
    """The best backend this card can run and that really loads; PyTorch's own attention otherwise."""
    import torch
    if torch.cuda.is_available():
        candidates = backends_for(torch.cuda.get_device_capability())
    else:
        candidates = list(MIN_CAPABILITY)
    for name in candidates:
        try:
            importlib.import_module(_BACKEND_MODULES[name])
            return name
        except Exception as e:   # a missing wheel, or a DLL that will not load on this card or driver
            print(f"[ATTENTION] {name} cannot be used: {type(e).__name__}: {e}")
    return 'sdpa'

def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    else:
        BACKEND = detect_best_backend()
    
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")
    print("Please wait...")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug