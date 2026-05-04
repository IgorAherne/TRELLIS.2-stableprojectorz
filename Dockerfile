# TRELLIS.2 — GPU-accelerated 3D generative AI
# Requires: NVIDIA Container Toolkit on the host
# Build: docker compose build
# Run:   docker compose up

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ── Environment ────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Compile all CUDA extensions for Pascal through Hopper in one shot.
# This keeps the image portable — no recompile needed per GPU generation.
ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9;9.0+PTX"

# HuggingFace model cache — points at the volume mount defined in docker-compose
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Runtime tuning (mirrors what app.py sets at startup)
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.65,expandable_segments:True"
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV TORCHDYNAMO_DISABLE=1

# ── System dependencies ────────────────────────────────────────────────────────
# Retries + redirect the flaky security mirror to the stable archive mirror
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
 && sed -i 's|http://security.ubuntu.com|http://archive.ubuntu.com|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libjpeg-dev \
    libgl1 \
    libglib2.0-0 \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ── PyTorch ───────────────────────────────────────────────────────────────────
# Separate layer — heavy download, rarely changes
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Core Python dependencies ──────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh \
    transformers accelerate huggingface_hub \
    "gradio==6.0.1" tensorboard pandas lpips \
    zstandard kornia timm \
    fastapi "uvicorn[standard]" \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Use stock Pillow — pillow-simd is incompatible with Gradio 6 (missing HAVE_WEBPANIM)
RUN pip install --no-cache-dir --upgrade Pillow

# ── CUDA extensions ───────────────────────────────────────────────────────────
# Each extension is its own layer so a single re-clone doesn't bust everything.

# nvdiffrast — NVIDIA differentiable rasterizer
RUN git clone -b v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
 && pip install --no-cache-dir /tmp/nvdiffrast --no-build-isolation \
 && rm -rf /tmp/nvdiffrast

# nvdiffrec — PBR material renderer (IgorAherne fork fixes garbage-init structs)
RUN git clone -b renderutils --depth 1 https://github.com/IgorAherne/nvdiffrec.git /tmp/nvdiffrec \
 && pip install --no-cache-dir /tmp/nvdiffrec --no-build-isolation \
 && rm -rf /tmp/nvdiffrec

# CuMesh — CUDA-accelerated mesh processing
RUN git clone --recursive --depth 1 https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh \
 && pip install --no-cache-dir /tmp/CuMesh --no-build-isolation \
 && rm -rf /tmp/CuMesh

# FlexGEMM — sparse convolution (Triton backend)
RUN git clone --recursive --depth 1 https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM \
 && pip install --no-cache-dir /tmp/FlexGEMM --no-build-isolation \
 && rm -rf /tmp/FlexGEMM

# flash-attn — preferred attention backend for Ampere+ (SM >= 8.0).
# Compilation against the full TORCH_CUDA_ARCH_LIST takes 30-60 min.
# Falls back gracefully to xformers at runtime if this step fails.
RUN pip install --no-cache-dir flash-attn==2.7.3 \
 || echo "[WARN] flash-attn build failed — xformers will be used at runtime"

# Patch flex_gemm so its triton kernels don't crash under Triton 3.x.
# pipeline_worker.py already provides a pure-PyTorch fallback for kernels.triton;
# this just stops the import from exploding before that patch can run.
RUN python - <<'EOF'
import pathlib
f = pathlib.Path('/usr/local/lib/python3.11/dist-packages/flex_gemm/kernels/__init__.py')
old = 'from . import triton'
new = 'try:\n    from . import triton\nexcept Exception:\n    pass  # Triton 3.x incompatible; fallback applied by pipeline_worker'
f.write_text(f.read_text().replace(old, new))
print('flex_gemm kernels patched')
EOF

# ── o-voxel (local CUDA extension) ───────────────────────────────────────────
# Copied before the full app COPY so this compilation layer stays cached
# independently of application source changes.
COPY o-voxel /tmp/o-voxel
# Eigen submodule is not initialized in the repo — fetch it directly
RUN git clone --depth 1 https://gitlab.com/libeigen/eigen.git /tmp/o-voxel/third_party/eigen \
 && pip install --no-cache-dir /tmp/o-voxel --no-build-isolation \
 && rm -rf /tmp/o-voxel

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# Runtime directories (volumes can overlay these at run time)
RUN mkdir -p tmp temp/current_generation models

EXPOSE 8080 7960

RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
