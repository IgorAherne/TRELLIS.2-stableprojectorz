# api_spz/routes/generation.py
import logging
import time
import traceback
import threading
import ctypes
from typing import Dict, Optional, List
import asyncio
import io
import base64
import os
from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from PIL import Image
import torch
import o_voxel

from api_spz.core.exceptions import CancelledException
from api_spz.core.files_manage import file_manager
from api_spz.core.state_manage import state
from api_spz.core.models_pydantic import (
    GenerationArgForm,
    GenerationResponse,
    TaskStatus,
    StatusResponse
)


router = APIRouter()

logger = logging.getLogger("trellis2_api")

# CHANGED: Thread-based cancellation instead of asyncio.Event
_worker_thread_id = None  # Tracks the running worker thread for cancellation

# A single lock to ensure only one generation at a time
generation_lock = asyncio.Lock()
def is_generation_in_progress() -> bool:
    return generation_lock.locked()


# A single dictionary holding "current generation" metadata
current_generation = {
    "status": TaskStatus.FAILED,  # default
    "progress": 0,
    "message": "",
    "model_url": None,
}


def reset_current_generation():
    current_generation["status"] = TaskStatus.PROCESSING
    current_generation["progress"] = 0
    current_generation["message"] = ""
    current_generation["model_url"] = None


def update_current_generation(
    status: Optional[TaskStatus] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
):
    if status is not None:
        current_generation["status"] = status
    if progress is not None:
        current_generation["progress"] = progress
    if message is not None:
        current_generation["message"] = message


async def cleanup_generation_files(keep_model: bool = False):
    file_manager.cleanup_generation_files(keep_model=keep_model)


def _validate_params(image_sources, arg: GenerationArgForm):
    """Validate incoming parameters before generation."""
    if not image_sources or len(image_sources) == 0:
        raise HTTPException(400, "No input images provided")
    if not (1 <= arg.guidance_scale <= 10):
        raise HTTPException(status_code=400, detail="Guidance scale must be between 1 and 10")
    if not (1 <= arg.num_inference_steps <= 50):
        raise HTTPException(status_code=400, detail="Inference steps must be between 1 and 50")
    if not (10000 <= arg.decimation_target <= 500000):
        raise HTTPException(status_code=400, detail="mesh_simplify must be between 10 and 500")
    if arg.output_format not in ["glb"]:
        raise HTTPException(status_code=400, detail="Only GLB output format is supported")


async def _get_image(file: Optional[UploadFile], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        try:
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
    else:
        try:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Issue opening the image file: {str(e)}")
    return pil_image


async def _load_images_into_list(
    files: Optional[List[UploadFile]] = None,
    images_base64: Optional[List[str]] = None,
) -> List[Image.Image]:
    all_images = []
    files = files or []
    images_base64 = images_base64 or []
    for b64_str in images_base64:
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]
        img = await _get_image(file=None, image_base64=b64_str)
        all_images.append(img)
    for f in files:
        img = await _get_image(file=f, image_base64=None)
        all_images.append(img)

    if not all_images:
        raise HTTPException(status_code=400, detail="No images provided (files or base64).")

    return all_images


async def _run_pipeline_generate_and_export(image: Image.Image, arg: GenerationArgForm):
    """Run Trellis 2 pipeline: generate mesh, then export to GLB."""
    def worker():
        # CHANGED: Register this thread for cancellation
        global _worker_thread_id
        _worker_thread_id = threading.current_thread().ident
        try:
            pipeline = state.pipeline

            torch.cuda.empty_cache()

            # Map simplified API params to Trellis 2's 3-stage sampler params.
            # guidance_scale controls sparse structure guidance (the most impactful stage).
            # num_inference_steps controls the step count for all three stages uniformly.
            # Other params use hardcoded defaults matching the Gradio app's tuned values.
            ss_params = {
                "steps": arg.num_inference_steps,
                "guidance_strength": arg.guidance_scale,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            }
            shape_params = {
                "steps": arg.num_inference_steps,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            }
            tex_params = {
                "steps": arg.num_inference_steps,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            }

            logger.info(f"Starting Trellis 2 generation (seed={arg.seed}, steps={arg.num_inference_steps}, guidance={arg.guidance_scale}, pipeline={arg.pipeline_type})")

            # Run the full pipeline (shape + texture in one pass)
            mesh = pipeline.run(
                image,
                seed=arg.seed,
                preprocess_image=True,
                sparse_structure_sampler_params=ss_params,
                shape_slat_sampler_params=shape_params,
                tex_slat_sampler_params=tex_params,
                pipeline_type=arg.pipeline_type,
            )[0]

            logger.info("Pipeline generation complete, exporting to GLB...")

            # Simplify to nvdiffrast limit before GLB export
            mesh.simplify(16777216)

            # Export to GLB via o_voxel (remeshing + texture baking)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=arg.decimation_target,
                texture_size=arg.texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )

            model_path = file_manager.get_temp_path("model.glb")
            glb.export(str(model_path), extension_webp=True)

            logger.info(f"GLB exported to {model_path}")

            del mesh, glb
            torch.cuda.empty_cache()

            return model_path

        except CancelledException:
            logger.info("Generation was cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise e
        finally:
            _worker_thread_id = None
            torch.cuda.empty_cache()

    return await asyncio.to_thread(worker)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@router.get("/ping")
async def ping():
    """Root endpoint to check server status."""
    busy = is_generation_in_progress()
    return {
        "status": "running",
        "message": "Trellis2 API is operational",
        "busy": busy
    }


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get status of the single current/last generation."""
    return StatusResponse(
        status=current_generation["status"],
        progress=current_generation["progress"],
        message=current_generation["message"],
        busy=is_generation_in_progress(),
    )


@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArgForm = Depends()
):
    """Generate a 3D model from a single image. Download GLB when complete."""
    print()  # empty line, for an easier read
    logger.info("Client asked to generate")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")

    start_time = time.time()
    reset_current_generation()
    try:
        _validate_params([file or image_base64], arg)

        image = await _get_image(file, image_base64)

        update_current_generation(status=TaskStatus.PROCESSING, progress=10, message="Generating 3D mesh and texture...")
        await _run_pipeline_generate_and_export(image, arg)
        update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete")

        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time
        logger.info(f"Generation completed in {duration:.2f} seconds")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"
        )
    except CancelledException:
        update_current_generation(status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@router.post("/generate_multi_no_preview", response_model=GenerationResponse)
async def generate_multi_no_preview(
    file_list: Optional[List[UploadFile]] = File(None),
    image_list_base64: Optional[List[str]] = Form(None),
    arg: GenerationArgForm = Depends(),
):
    """
    Generate a 3D model from images (Trellis 2 uses only the first image).
    """
    print()
    logger.info("Client asked to multi-view-generate (using first image only)")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")

    start_time = time.time()
    reset_current_generation()

    try:
        _validate_params(file_list or image_list_base64, arg)
        images = await _load_images_into_list(file_list, image_list_base64)

        # Trellis 2 uses a single image — take the first one
        image = images[0]

        update_current_generation(status=TaskStatus.PROCESSING, progress=10, message="Generating 3D mesh and texture...")
        await _run_pipeline_generate_and_export(image, arg)
        update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete")

        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time
        logger.info(f"Generation completed in {duration:.2f} seconds")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"
        )
    except CancelledException:
        update_current_generation(status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@router.post("/generate", response_model=GenerationResponse)
async def process_ui_generation_request(data: Dict):
    """Process generation request from the StableProjectorz UI panel."""
    try:
        arg = GenerationArgForm(
            seed=int(data["seed"]),
            guidance_scale=float(data["guidance_scale"]),
            num_inference_steps=int(data["num_inference_steps"]),
            resolution=int(data["resolution"]),
            mesh_simplify=int(data["mesh_simplify"]),
            apply_texture=data["apply_texture"],
            texture_size=2048,
            output_format="glb",
        )
        images_base64 = data["single_multi_img_input"]
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")

        response = await generate_multi_no_preview(
            file_list=None,
            image_list_base64=images_base64,
            arg=arg,
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing UI generation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# "make_meshes_and_tex" — Trellis 2 always generates mesh with texture
@router.get("/info/supported_operations")
async def get_supported_operation_types():
    return ["make_meshes_and_tex"]


# CHANGED: Real thread-level cancellation via PyThreadState_SetAsyncExc
@router.post("/interrupt")
async def interrupt_generation():
    """Interrupt the current generation if one is in progress."""
    logger.info("Client cancelled the generation.")
    if not is_generation_in_progress():
        return {"status": "no_generation_in_progress"}

    thread_id = _worker_thread_id
    if thread_id is not None:
        # Inject CancelledException into the worker thread.
        # It fires at the next Python bytecode boundary (between sampling steps, ~1-2s).
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.py_object(CancelledException)
        )
        if ret == 0:
            logger.warning("Thread ID not found for cancellation (thread may have finished)")
        elif ret > 1:
            # Shouldn't happen - clear it to avoid instability
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
            logger.error("PyThreadState_SetAsyncExc affected multiple threads - cleared")
        else:
            logger.info(f"CancelledException injected into worker thread {thread_id}")
    else:
        logger.warning("Generation lock is held but no worker thread ID registered")

    return {"status": "interrupt_requested"}


@router.get("/download/model")
async def download_model():
    """Download final 3D model (GLB)."""
    logger.info("Client is downloading a model.")
    model_path = file_manager.get_temp_path("model.glb")
    if not model_path.exists():
        logger.error("mesh not found")
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename="model.glb"
    )


@router.get("/download/spz-ui-layout/generation-3d-panel")
async def get_generation_panel_layout():
    """Return the UI layout for the generation panel."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'layout_generation_3d_panel.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # By using Response(content=content, media_type="text/plain; charset=utf-8")
            # - Bypass the automatic JSON encoding
            # - Explicitly tell the client this is plain text (not JSON)
            # - Ensure proper UTF-8 encoding is maintained
            # This way Unity receives the layout text exactly as it appears in the file.
            return Response(content=content, media_type="text/plain; charset=utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Layout file not found")