# File: api_spz/routes/texturing.py
# api_spz/routes/texturing.py
"""
Shape-conditioned PBR texturing over HTTP: send a GLB and an image, get the same GLB back
with new base colour + metallic/roughness maps (Trellis2TexturingPipeline.retexture_glb).

Shares the generation routes' lock, /status, /interrupt and /download/model, so a client
drives it exactly like /generate_no_preview: POST, poll /status while it blocks, then
GET /download/model. The texturing pipeline is built on first use from the already-loaded
image-to-3D pipeline (shared models; only the shape encoder, ~0.7 GB, is loaded new).
"""
import asyncio
import base64
import io
import logging
import threading
import time
import traceback
from typing import Optional

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from api_spz.core.exceptions import CancelledException
from api_spz.core.files_manage import file_manager
from api_spz.core.models_pydantic import TaskStatus
from api_spz.core.state_manage import state
from api_spz.routes import generation as gen

router = APIRouter()
logger = logging.getLogger("trellis2_api")

_texturing = None
_texturing_lock = threading.Lock()


def texturing_pipeline():
    """The texturing pipeline, built once from the loaded image pipeline."""
    global _texturing
    with _texturing_lock:
        if _texturing is None:
            from trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline
            logger.info("Building the texturing pipeline (shape encoder + shared models)...")
            _texturing = Trellis2TexturingPipeline.from_image_pipeline(state.pipeline)
            logger.info("Texturing pipeline ready")
        return _texturing


async def _read_upload(file: Optional[UploadFile], b64: Optional[str], what: str) -> bytes:
    if b64:
        try:
            if "base64," in b64:
                b64 = b64.split("base64,")[1]
            return base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 {what}: {e}")
    if file is None:
        raise HTTPException(status_code=400, detail=f"No {what} provided")
    return await file.read()


@router.post("/texture_no_preview")
async def texture_no_preview(
    mesh: Optional[UploadFile] = File(None),
    mesh_base64: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    seed: int = Form(42),
    resolution: int = Form(1024),
    num_inference_steps: int = Form(12),
    tex_guidance_strength: float = Form(1.0),
    tex_rescale_t: float = Form(3.0),
    texture_size: int = Form(0),
    max_texture_size: int = Form(4096),
    uv_mode: str = Form("auto"),
):
    """
    Re-texture a GLB from an image. `mesh` is the .glb, `file` the image (as for
    /generate_no_preview). resolution 512 | 1024; texture_size 0 keeps each material's
    current size; uv_mode auto | keep | new. Returns the usual envelope plus `report`
    (per-material UV decision, coverage, texture size) once the GLB is ready at
    /download/model.
    """
    print()
    logger.info("Client asked to texture a mesh")
    try:
        await asyncio.wait_for(gen.generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")

    start = time.time()
    gen.reset_current_generation()
    try:
        if resolution not in (512, 1024):
            raise HTTPException(status_code=400, detail="resolution must be 512 or 1024")
        if not (1 <= num_inference_steps <= 50):
            raise HTTPException(status_code=400, detail="Inference steps must be between 1 and 50")
        if uv_mode not in ("auto", "keep", "new"):
            raise HTTPException(status_code=400, detail="uv_mode must be auto, keep or new")
        if texture_size and not (256 <= texture_size <= 8192):
            raise HTTPException(status_code=400, detail="texture_size must be 0 or 256-8192")
        glb = await _read_upload(mesh, mesh_base64, "mesh")
        img_bytes = await _read_upload(file, image_base64, "image")
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Issue opening the image: {e}")

        params = {
            "steps": num_inference_steps,
            "guidance_strength": tex_guidance_strength,
            "guidance_rescale": 0.5 if tex_guidance_strength > 1.0 else 0.0,
            "rescale_t": tex_rescale_t,
        }

        def progress(pct, msg):
            gen.update_current_generation(status=TaskStatus.PROCESSING, progress=pct, message=msg)

        def worker():
            gen._worker_thread_id = threading.current_thread().ident  # /interrupt reaches us
            try:
                progress(2, "Preparing the texturing pipeline")
                pipe = texturing_pipeline()
                torch.cuda.empty_cache()
                out, report = pipe.retexture_glb(
                    glb, image, seed=seed, tex_slat_sampler_params=params, resolution=resolution,
                    texture_size=texture_size, max_texture_size=max_texture_size, uv_mode=uv_mode,
                    progress=progress)
                path = file_manager.get_temp_path("model.glb")
                with open(path, "wb") as f:
                    f.write(out)
                return report
            finally:
                gen._worker_thread_id = None
                torch.cuda.empty_cache()

        progress(1, "Texturing the mesh...")
        report = await asyncio.to_thread(worker)
        gen.update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Texturing complete")
        await gen.cleanup_generation_files(keep_model=True)
        logger.info(f"Texturing completed in {time.time() - start:.2f} seconds")
        return {
            "status": TaskStatus.COMPLETE,
            "progress": 100,
            "message": "Texturing complete",
            "model_url": "/download/model",
            "report": report,
        }
    except CancelledException:
        gen.update_current_generation(status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await gen.cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Texturing cancelled by user")
    except HTTPException as e:
        gen.update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e.detail))
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        gen.update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await gen.cleanup_generation_files()
        # A bad input (no UVs with uv_mode=keep, unsupported GLB) is the request's fault.
        raise HTTPException(status_code=400 if isinstance(e, ValueError) else 500, detail=str(e))
    finally:
        gen.generation_lock.release()


@router.get("/info/texturing")
async def texturing_info():
    """Lets a client check this server can texture before it sends a job."""
    return {"texturing": True, "resolutions": [512, 1024], "uv_modes": ["auto", "keep", "new"]}
