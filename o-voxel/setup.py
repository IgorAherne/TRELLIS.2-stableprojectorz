# File: o-voxel/setup.py
# o-voxel/setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Compiler Flags Configuration ---

# Windows-specific C++ flags
# /bigobj is crucial for heavy template libraries like Eigen on Windows
if os.name == 'nt':
    cxx_flags = ['/O2', '/std:c++17', '/bigobj', '/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS']
else:
    cxx_flags = ['-O3', '-std=c++17']

# NVCC Flags (CUDA Compiler)
nvcc_flags = [
    '-O3',
    '--use_fast_math', 
    '--expt-relaxed-constexpr',
    '--allow-unsupported-compiler', # Helps with newer VS versions
    '-std=c++17'
]

# --- Architecture Targeting ---
# This ensures the wheel works on GTX 10xx (Pascal) through RTX 50xx (Blackwell/Future)
# We bake the PTX code in so it works without the user needing nvcc.
arch_flags = [
    '-gencode=arch=compute_61,code=sm_61',      # GTX 10 Series (Pascal)
    '-gencode=arch=compute_75,code=sm_75',      # RTX 20 Series (Turing)
    '-gencode=arch=compute_86,code=sm_86',      # RTX 30 Series (Ampere)
    '-gencode=arch=compute_89,code=sm_89',      # RTX 40 Series (Ada)
    '-gencode=arch=compute_90,code=sm_90',      # H100 / RTX 50 Series (Hopper/Blackwell base)
    '-gencode=arch=compute_90,code=compute_90'  # PTX for future compatibility
]
nvcc_flags.extend(arch_flags)

# Explicitly set the list for PyTorch to avoid auto-detection issues during build
os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1;7.5;8.6;8.9;9.0'

setup(
    name="o_voxel",
    version="0.1.0", # Added versioning
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                # Hashmap functions
                "src/hash/hash.cu",
                # Convert functions
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                ## Serialization functions
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                # IO functions
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                # Rasterization functions
                "src/rasterize/rasterize.cu",
                # main
                "src/ext.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/eigen"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)