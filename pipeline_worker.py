# pipeline_worker.py
# Runs the TRELLIS.2 pipeline in a separate process to avoid GIL contention with Gradio.
import multiprocessing as mp
import traceback
import os
import sys


def _apply_patches():
    """Apply Windows compatibility patches for flex_gemm."""
    import torch

    try:
        import flex_gemm.ops.spconv as spconv
        from flex_gemm.ops.spconv import Algorithm
        # Force the algorithm to EXPLICIT_GEMM
        # This bypasses the 'kernels.triton' error by using standard Torch Matrix Multiplication
        spconv.ALGORITHM = Algorithm.EXPLICIT_GEMM
        print("[WORKER] flex_gemm EXPLICIT_GEMM patch applied.")
    except ImportError:
        print("[WORKER] Could not patch flex_gemm spconv.")
    except Exception as e:
        print(f"[WORKER] flex_gemm spconv patch failed: {e}")

    try:
        import flex_gemm.kernels as _fgk
        if not hasattr(_fgk, 'triton'):
            class _TritonFallback:
                @staticmethod
                def indice_weighed_sum_fwd(feats, indices, weights):
                    N = feats.shape[0]
                    idx = indices.long().clamp(min=0, max=N - 1)  # [M, 8]
                    gathered = feats[idx]  # [M, 8, C]
                    return (gathered * weights.unsqueeze(-1)).sum(dim=1)  # [M, C]

                @staticmethod
                def indice_weighed_sum_bwd_input(grad_output, indices, weights, N):
                    M, C = grad_output.shape
                    idx = indices.long().clamp(min=0, max=N - 1)
                    weighted_grad = grad_output.unsqueeze(1) * weights.unsqueeze(-1)  # [M, 8, C]
                    grad_feats = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)
                    grad_feats.scatter_add_(0, idx.reshape(-1, 1).expand(-1, C), weighted_grad.reshape(-1, C))
                    return grad_feats

            _fgk.triton = _TritonFallback()
            print("[WORKER] flex_gemm Triton fallback patch applied.")
    except ImportError:
        print("[WORKER] Could not patch flex_gemm triton fallback.")
    except Exception as e:
        print(f"[WORKER] flex_gemm triton patch failed: {e}")


def _worker_main(cmd_queue, result_queue):
    """Worker process main loop. Owns the pipeline and all GPU resources."""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    _apply_patches()

    import torch
    import cv2
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    from trellis2.modules.sparse import SparseTensor
    import o_voxel

    print("[WORKER] Loading pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()

    envmap = {
        'forest': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'sunset': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'courtyard': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
    }

    print("[WORKER] Pipeline ready.", flush=True)
    result_queue.put({"status": "ready"})

    while True:
        cmd = cmd_queue.get()
        action = cmd["action"]

        if action == "shutdown":
            print("[WORKER] Shutting down.")
            break

        elif action == "preprocess":
            try:
                image = pipeline.preprocess_image(cmd["image"])
                result_queue.put({"status": "ok", "image": image})
            except Exception as e:
                traceback.print_exc()
                result_queue.put({"status": "error", "error": str(e)})

        elif action == "generate":
            try:
                outputs, latents = pipeline.run(
                    cmd["image"],
                    seed=cmd["seed"],
                    preprocess_image=False,
                    sparse_structure_sampler_params=cmd["ss_params"],
                    shape_slat_sampler_params=cmd["shape_params"],
                    tex_slat_sampler_params=cmd["tex_params"],
                    pipeline_type=cmd["pipeline_type"],
                    return_latent=True,
                )
                mesh = outputs[0]
                mesh.simplify(16777216)  # nvdiffrast limit
                images = render_utils.render_snapshot(
                    mesh, resolution=1024, r=2, fov=36,
                    nviews=cmd["nviews"], envmap=envmap
                )
                shape_slat, tex_slat, res = latents
                state = {
                    'shape_slat_feats': shape_slat.feats.cpu().numpy(),
                    'tex_slat_feats': tex_slat.feats.cpu().numpy(),
                    'coords': shape_slat.coords.cpu().numpy(),
                    'res': res,
                }
                torch.cuda.empty_cache()
                result_queue.put({"status": "ok", "state": state, "images": images})
            except Exception as e:
                traceback.print_exc()
                result_queue.put({"status": "error", "error": str(e)})

        elif action == "extract_glb":
            try:
                state = cmd["state"]
                shape_slat = SparseTensor(
                    feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
                    coords=torch.from_numpy(state['coords']).cuda(),
                )
                tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
                res = state['res']
                mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=pipeline.pbr_attr_layout,
                    grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=cmd["decimation_target"],
                    texture_size=cmd["texture_size"],
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    use_tqdm=True,
                )
                glb.export(cmd["glb_path"], extension_webp=True)
                torch.cuda.empty_cache()
                result_queue.put({"status": "ok", "glb_path": cmd["glb_path"]})
            except Exception as e:
                traceback.print_exc()
                result_queue.put({"status": "error", "error": str(e)})


class PipelineWorker:
    """Manages a subprocess that owns the GPU pipeline."""

    def __init__(self):
        ctx = mp.get_context('spawn')  # 'spawn' is required for CUDA on Windows
        self.cmd_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(target=_worker_main, args=(self.cmd_queue, self.result_queue))
        self.process.daemon = True
        print("[INFO] Starting pipeline worker process...")
        self.process.start()
        msg = self.result_queue.get(timeout=300)
        assert msg["status"] == "ready", f"Worker failed to start: {msg}"
        print("[INFO] Pipeline worker ready.")

    def preprocess(self, image):
        self.cmd_queue.put({"action": "preprocess", "image": image})
        result = self.result_queue.get(timeout=120)
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["image"]

    def generate(self, image, seed, ss_params, shape_params, tex_params, pipeline_type, nviews):
        self.cmd_queue.put({
            "action": "generate",
            "image": image,
            "seed": seed,
            "ss_params": ss_params,
            "shape_params": shape_params,
            "tex_params": tex_params,
            "pipeline_type": pipeline_type,
            "nviews": nviews,
        })
        result = self.result_queue.get(timeout=600)
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["state"], result["images"]

    def extract_glb(self, state, decimation_target, texture_size, glb_path):
        self.cmd_queue.put({
            "action": "extract_glb",
            "state": state,
            "decimation_target": decimation_target,
            "texture_size": texture_size,
            "glb_path": glb_path,
        })
        result = self.result_queue.get(timeout=600)
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["glb_path"]

    def shutdown(self):
        try:
            self.cmd_queue.put({"action": "shutdown"})
            self.process.join(timeout=10)
        except Exception:
            self.process.kill()