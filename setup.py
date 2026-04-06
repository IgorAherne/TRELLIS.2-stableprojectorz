# setup.py
# File: setup.py
from setuptools import setup, find_packages

# Basic dependencies, corresponds to the --basic flag in your script.
# NOTE: PyTorch is a core dependency but should be installed manually before
# running this setup. See the installation instructions for details.
install_requires = [
    'imageio',
    'imageio-ffmpeg',
    'tqdm',
    'easydict',
    'opencv-python-headless',
    'ninja',
    'trimesh',
    'transformers',
    'gradio==6.0.1', # Note: This version seems unusual, please verify if it's correct.
    'tensorboard',
    'pandas',
    'lpips',
    'zstandard',
    # pillow-simd is a faster drop-in replacement for Pillow.
    # The original script had a system dependency for Linux ('libjpeg-dev').
    # On Windows, pre-built wheels for pillow-simd are usually available.
    'pillow-simd',
    'kornia',
    'timm',
    'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8',
]

# Optional dependencies, mapped from the flags in the original script.
# You can install these using 'pip install .[extra_name]'.
# For example: pip install .[flash-attn,cumesh]
extras_require = {
    # Corresponds to --flash-attn
    # Note: Installing flash-attn on Windows can be challenging. If this fails,
    # you may need to build it from source, which can be complex.
    'flash-attn': ['flash-attn==2.7.3'],

    # Corresponds to --nvdiffrast
    # The original script used '--no-build-isolation'. If you encounter build issues,
    # you may need to run: pip install .[nvdiffrast] --no-build-isolation
    'nvdiffrast': ['nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0'],

    # Corresponds to --nvdiffrec
    'nvdiffrec': ['nvdiffrec @ git+https://github.com/IgorAherne/nvdiffrec.git@renderutils'], #Igor fixed garbage-init structs in cpp file.

    # Corresponds to --cumesh
    'cumesh': ['CuMesh @ git+https://github.com/JeffreyXiang/CuMesh.git'],

    # Corresponds to --flexgemm
    'flexgemm': ['FlexGEMM @ git+https://github.com/JeffreyXiang/FlexGEMM.git'],
}

# The 'o-voxel' dependency from your script is a local directory and should be
# installed separately. See the instructions for how to do this.

# An 'all' extra to install all optional dependencies at once.
# Note that this does not include 'o-voxel'.
all_deps = []
for group in extras_require.values():
    all_deps.extend(group)
extras_require['all'] = all_deps


setup(
    name='trellis2', # Name taken from the conda environment in your script
    version='0.1.0',
    description="A project setup converted from a shell script.",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.10, <3.12', # As you requested Python 11 (3.11)
)