# trellis2/modules/sparse/config.py
from typing import *

CONV = 'flex_gemm' 
DEBUG = False
ATTN = None  # resolved in __from_env

def _detect_best_backend():
    """Pick the best available sparse attention backend based on GPU compute capability."""
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm >= 80:
            return 'flash_attn'
        else:
            return 'xformers'
    return 'flash_attn'

def __from_env():
    import os
    
    global CONV
    global DEBUG
    global ATTN
    
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']:
        ATTN = env_sparse_attn_backend
    else:
        ATTN = _detect_best_backend()
        
    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")
        

__from_env()
    

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn']):
    global ATTN
    ATTN = backend