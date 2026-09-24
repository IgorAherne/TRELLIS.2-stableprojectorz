# api_spz/core/models_pydantic.py
import re
from enum import Enum
from typing import Optional, Dict
from fastapi import Form, HTTPException
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PROCESSING = "PROCESSING"
    PREVIEW_READY = "PREVIEW_READY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class GenerationArgForm:
    def __init__(
        self,
        seed: int = Form(1234),
        guidance_scale: float = Form(7.5),
        num_inference_steps: int = Form(12),
        resolution: int = Form(1024),
        mesh_simplify: int = Form(50),
        apply_texture: bool = Form(True),
        texture_size: int = Form(2048),
        tex_rescale_t: float = Form(3.0),
        tex_guidance_strength: float = Form(1.0),
        output_format: str = Form("glb"),
        chroma_key: str = Form(""),          # '' | 'auto' | '#rrggbb' -> flat background becomes alpha
        chroma_tolerance: int = Form(24),    # RGB distance counted as background
        chroma_shadows: bool = Form(True),   # also key the subject's shadow on the backdrop
    ):
        self.chroma_key = chroma_key.strip().lower()
        # Checked here rather than in the route: FastAPI resolves this form before the
        # handler's try/except, so a bad value is answered with a 400 instead of a 500.
        if self.chroma_key not in ("", "auto") and not re.fullmatch(r"#?[0-9a-f]{6}", self.chroma_key):
            raise HTTPException(status_code=400, detail="chroma_key must be '', 'auto' or '#rrggbb'")
        self.chroma_tolerance = chroma_tolerance
        self.chroma_shadows = chroma_shadows
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        # mesh_simplify arrives as thousands (e.g. 50 => 50,000 faces decimation target)
        self.decimation_target = mesh_simplify * 1000
        self.apply_texture = apply_texture
        self.texture_size = texture_size
        self.tex_rescale_t = tex_rescale_t
        self.tex_guidance_strength = tex_guidance_strength
        self.output_format = output_format
        # Snap resolution to nearest valid value and map to pipeline_type
        if resolution <= 768:
            self.pipeline_type = '512'
        elif resolution <= 1280:
            self.pipeline_type = '1024_cascade'
        else:
            self.pipeline_type = '1536_cascade'


class GenerationResponse(BaseModel):
    status: TaskStatus
    progress: int = 0
    message: str = ""
    model_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: TaskStatus
    progress: int
    message: str
    busy: bool