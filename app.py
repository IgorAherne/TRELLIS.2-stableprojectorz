# File: app.py
# app.py
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import gradio as gr

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import shutil
from typing import *
import numpy as np
from PIL import Image

import base64
import io
import atexit, signal

from pipeline_worker import PipelineWorker
from trellis2.modules.sparse import SparseTensor
import torch

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}

/* Previewer */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: 722px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 300px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.2;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

/* Row 1: Display Modes */
.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.1); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.1);
}

/* Row 2: Display Image */
.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 400px;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible {
    display: block;
}

/* Row 3: Custom HTML Slider */
.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 400px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

/* Overwrite Previewer Block Style */
.gradio-container .padded:has(.previewer-container) {
    padding: 0 !important;
}

.gradio-container:has(.previewer-container) [data-testid="block-label"] {
    position: absolute;
    top: 0;
    left: 0;
}
"""


head = """
<script>
    function refreshView(mode, step) {
        // 1. Find current mode and step
        const allImgs = document.querySelectorAll('.previewer-main-image');
        for (let i = 0; i < allImgs.length; i++) {
            const img = allImgs[i];
            if (img.classList.contains('visible')) {
                const id = img.id;
                const [_, m, s] = id.split('-');
                if (mode === -1) mode = parseInt(m.slice(1));
                if (step === -1) step = parseInt(s.slice(1));
                break;
            }
        }
        
        // 2. Hide ALL images
        // We select all elements with class 'previewer-main-image'
        allImgs.forEach(img => img.classList.remove('visible'));

        // 3. Construct the specific ID for the current state
        // Format: view-m{mode}-s{step}
        const targetId = 'view-m' + mode + '-s' + step;
        const targetImg = document.getElementById(targetId);

        // 4. Show ONLY the target
        if (targetImg) {
            targetImg.classList.add('visible');
        }

        // 5. Update Button Highlights
        const allBtns = document.querySelectorAll('.mode-btn');
        allBtns.forEach((btn, idx) => {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }
    
    // --- Action: Switch Mode ---
    function selectMode(mode) {
        refreshView(mode, -1);
    }
    
    // --- Action: Slider Change ---
    function onSliderChange(val) {
        refreshView(-1, parseInt(val));
    }
</script>
"""


empty_html = f"""
<div class="previewer-container">
    <svg style=" opacity: .5; height: var(--size-5); color: var(--body-text-color);"
    xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
</div>
"""


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    return worker.preprocess(image)


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    
    
def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    enable_python_profiling: bool,
    enable_torch_profiling: bool,
    enable_sync_hunter: bool,
    profiler_delay_sec: float,
    profiler_max_duration_sec: float,
    profiler_max_events: int,
    req: gr.Request,
) -> str:
    try:
        state, images = worker.generate(
            image=image,
            seed=seed,
            ss_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type={
                "512": "512",
                "1024": "1024_cascade",
                "1536": "1536_cascade",
            }[resolution],
            nviews=STEPS,
            profiling={
                "enable_python": enable_python_profiling,
                "enable_torch": enable_torch_profiling,
                "enable_sync_hunter": enable_sync_hunter,
                "delay_sec": profiler_delay_sec,
                "max_duration_sec": profiler_max_duration_sec,
                "max_events": profiler_max_events,
            },
        )
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return {}, empty_html

    # --- HTML Construction ---
    # The Stack of 48 Images
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            # ID Naming Convention: view-m{mode}-s{step}
            unique_id = f"view-m{m_idx}-s{s_idx}"
            
            # Logic: Only Mode 0, Step 0 is visible initially
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            
            # Image Source
            img_base64 = image_to_base64(Image.fromarray(images[mode['render_key']][s_idx]))
            
            # Render the Tag
            images_html += f"""
                <img id="{unique_id}" 
                     class="previewer-main-image {vis_class}" 
                     src="{img_base64}" 
                     loading="eager">
            """
    
    # Button Row HTML
    btns_html = ""
    for idx, mode in enumerate(MODES):        
        active_class = "active" if idx == DEFAULT_MODE else ""
        # Note: onclick calls the JS function defined in Head
        btns_html += f"""
            <img src="{mode['icon_base64']}" 
                 class="mode-btn {active_class}" 
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """
    
    # Assemble the full component
    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> - Click on the circular buttons to switch between different render modes.</p>
                <p>● <b>View Angle</b> - Drag the slider to change the view angle.</p>
            </div>
        </div>
        
        <!-- Row 1: Viewport containing 48 static <img> tags -->
        <div class="display-row">
            {images_html}
        </div>
        
        <!-- Row 2 -->
        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <!-- Row 3: Slider -->
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    return state, full_html


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    worker.extract_glb(state, decimation_target, texture_size, glb_path)
    return glb_path, glb_path



def create_input_panel():
    with gr.Column(scale=1, min_width=360):
        image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
        
        resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
        seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
        randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
        decimation_target = gr.Slider(100000, 1000000, label="Decimation Target", value=500000, step=10000)
        texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)
        
        generate_btn = gr.Button("Generate")
            
        with gr.Accordion(label="Advanced Settings", open=False):                
            gr.Markdown("Stage 1: Sparse Structure Generation")
            with gr.Row():
                ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
            gr.Markdown("Stage 2: Shape Generation")
            with gr.Row():
                shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
            gr.Markdown("Stage 3: Material Generation")
            with gr.Row():
                tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)

        with gr.Accordion(label="Debugging & Profiling", open=False):
            gr.Markdown("Performance profiles and sync logs will be stored in `./tmp/profiling`.")
            with gr.Row():
                enable_python_profiling = gr.Checkbox(label="Enable Python Profiler", value=False)
                enable_torch_profiling = gr.Checkbox(label="Enable PyTorch Profiler", value=False)
                enable_sync_hunter = gr.Checkbox(label="Enable CUDA Sync Hunter", value=False)
            with gr.Row():
                profiler_delay_sec = gr.Slider(
                    label="Start Delay (seconds)",
                    minimum=0, maximum=300, value=80, step=5,
                    info="Wait X seconds before starting to record."
                )
                profiler_max_duration_sec = gr.Slider(
                    label="Recording Duration (seconds)",
                    minimum=0, maximum=60, value=3, step=1,
                    info="Stop recording after X seconds (0 = unlimited)."
                )
            profiler_max_events = gr.Slider(
                label="Max Events (Datapoints)",
                minimum=0, maximum=5000, value=300, step=50,
                info="Stop recording after N operations (0 = unlimited)."
            )

    return {
        "image_prompt": image_prompt,
        "resolution": resolution,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "decimation_target": decimation_target,
        "texture_size": texture_size,
        "generate_btn": generate_btn,
        "ss_guidance_strength": ss_guidance_strength,
        "ss_guidance_rescale": ss_guidance_rescale,
        "ss_sampling_steps": ss_sampling_steps,
        "ss_rescale_t": ss_rescale_t,
        "shape_slat_guidance_strength": shape_slat_guidance_strength,
        "shape_slat_guidance_rescale": shape_slat_guidance_rescale,
        "shape_slat_sampling_steps": shape_slat_sampling_steps,
        "shape_slat_rescale_t": shape_slat_rescale_t,
        "tex_slat_guidance_strength": tex_slat_guidance_strength,
        "tex_slat_guidance_rescale": tex_slat_guidance_rescale,
        "tex_slat_sampling_steps": tex_slat_sampling_steps,
        "tex_slat_rescale_t": tex_slat_rescale_t,
        "enable_python_profiling": enable_python_profiling,
        "enable_torch_profiling": enable_torch_profiling,
        "enable_sync_hunter": enable_sync_hunter,
        "profiler_delay_sec": profiler_delay_sec,
        "profiler_max_duration_sec": profiler_max_duration_sec,
        "profiler_max_events": profiler_max_events,
    }

def create_preview_panel():
    with gr.Column(scale=10):
        with gr.Walkthrough(selected=0) as walkthrough:
            with gr.Step("Preview", id=0):
                preview_output = gr.HTML(empty_html, label="3D Asset Preview", show_label=True, container=True)
                extract_btn = gr.Button("Extract GLB", interactive=False)
            with gr.Step("Extract", id=1):
                glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                download_btn = gr.DownloadButton(label="Download GLB")
    
    return {
        "walkthrough": walkthrough,
        "preview_output": preview_output,
        "extract_btn": extract_btn,
        "glb_output": glb_output,
        "download_btn": download_btn
    }

def create_examples_panel(image_prompt):
    with gr.Column(scale=1, min_width=172):
        examples = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=18,
        )
    return examples


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2)
    * Upload an image (preferably with an alpha-masked foreground object) and click Generate to create a 3D asset.
    * Click Extract GLB to export and download the generated GLB file if you're satisfied with the result. Otherwise, try another time.
    """)
    
    with gr.Row():
        # 1. Input Panel
        inputs = create_input_panel()
        
        # 2. Preview Panel
        outputs = create_preview_panel()
        
        # 3. Examples Panel (Needs the image_prompt from inputs)
        create_examples_panel(inputs["image_prompt"])
                    
    output_buf = gr.State()

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    inputs["image_prompt"].upload(
        preprocess_image,
        inputs=[inputs["image_prompt"]],
        outputs=[inputs["image_prompt"]],
    )

    inputs["generate_btn"].click(
        get_seed,
        inputs=[inputs["randomize_seed"], inputs["seed"]],
        outputs=[inputs["seed"]],
    ).then(
        lambda: gr.Walkthrough(selected=0), outputs=outputs["walkthrough"]
    ).then(
        lambda: gr.Button(interactive=False), outputs=outputs["extract_btn"],
    ).then(
        image_to_3d,
        inputs=[
            inputs["image_prompt"], inputs["seed"], inputs["resolution"],
            inputs["ss_guidance_strength"], inputs["ss_guidance_rescale"], inputs["ss_sampling_steps"], inputs["ss_rescale_t"],
            inputs["shape_slat_guidance_strength"], inputs["shape_slat_guidance_rescale"], inputs["shape_slat_sampling_steps"], inputs["shape_slat_rescale_t"],
            inputs["tex_slat_guidance_strength"], inputs["tex_slat_guidance_rescale"], inputs["tex_slat_sampling_steps"], inputs["tex_slat_rescale_t"],
            inputs["enable_python_profiling"], inputs["enable_torch_profiling"], inputs["enable_sync_hunter"],
            inputs["profiler_delay_sec"], inputs["profiler_max_duration_sec"], inputs["profiler_max_events"],
        ],
        outputs=[output_buf, outputs["preview_output"]],
    ).then(
        lambda: gr.Button(interactive=True), outputs=outputs["extract_btn"],
    )
    
    outputs["extract_btn"].click(
        lambda: gr.Walkthrough(selected=1), outputs=outputs["walkthrough"]
    ).then(
        extract_glb,
        inputs=[output_buf, inputs["decimation_target"], inputs["texture_size"]],
        outputs=[outputs["glb_output"], outputs["download_btn"]],
    )
        

# Launch the Gradio app
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args, unknown = parser.parse_known_args()

    os.makedirs(TMP_DIR, exist_ok=True)

    # Construct ui components
    btn_img_base64_strs = {}
    for i in range(len(MODES)):
        icon = Image.open(MODES[i]['icon'])
        MODES[i]['icon_base64'] = image_to_base64(icon)

    worker = PipelineWorker()

    def _cleanup():
        worker.shutdown()

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (atexit._run_exitfuncs(), exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (atexit._run_exitfuncs(), exit(0)))
    
    import socket
    def _find_free_port(start=args.port, attempts=100):
        for p in range(start, start + attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex((args.host, p)) == 0:
                    continue
                if p!=args.port: 
                    print(f"\nBinding to port {p} because requested port {args.port} is already used by something else.")
                return p
        raise RuntimeError(f"No free port found in range {start}-{start+attempts}")

    port = _find_free_port(args.port)


    app, local_url, share_url = demo.launch(
        css=css, 
        head=head, 
        server_name=args.host, server_port=port,
        prevent_thread_lock=True)
    print("\n==============================================")
    print(f"Keep gpu-hungry 3D programs disabled during generation (avoid photoshop/blender/unity).")
    print(f"\nTo start work, open browser and enter  http://{args.host}:{port}  in the URL bar, like it's a website.")
    print("==============================================")
    demo.block_thread()