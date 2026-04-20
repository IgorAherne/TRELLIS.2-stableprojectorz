# api_spz/core/state_manage.py
import logging
import torch
from pathlib import Path

logger = logging.getLogger("trellis2_api")


def _apply_patches():
    """Apply Windows compatibility patches for flex_gemm."""
    try:
        import flex_gemm.ops.spconv as spconv
        from flex_gemm.ops.spconv import Algorithm
        # Force the algorithm to EXPLICIT_GEMM
        # This bypasses the 'kernels.triton' error by using standard Torch Matrix Multiplication
        spconv.ALGORITHM = Algorithm.EXPLICIT_GEMM
        print("[INIT] flex_gemm EXPLICIT_GEMM patch applied.")
    except ImportError:
        print("[INIT] Could not patch flex_gemm spconv.")
    except Exception as e:
        print(f"[INIT] flex_gemm spconv patch failed: {e}")

    try:
        import flex_gemm.kernels as _fgk
        if not hasattr(_fgk, 'triton'):
            class _TritonFallback:
                @staticmethod
                def indice_weighed_sum_fwd(feats, indices, weights):
                    N = feats.shape[0]
                    idx = indices.long().clamp(min=0, max=N - 1)  # [M, 8]
                    M_shape, K = idx.shape
                    C = feats.shape[-1]
                    # Accumulate sequentially to avoid a massive [M, K, C] memory spike
                    out = torch.zeros((M_shape, C), dtype=feats.dtype, device=feats.device)
                    for i in range(K):
                        out += feats[idx[:, i]] * weights[:, i].unsqueeze(-1)
                    return out

                @staticmethod
                def indice_weighed_sum_bwd_input(grad_output, indices, weights, N):
                    M, C = grad_output.shape
                    idx = indices.long().clamp(min=0, max=N - 1)
                    weighted_grad = grad_output.unsqueeze(1) * weights.unsqueeze(-1)  # [M, 8, C]
                    grad_feats = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)
                    grad_feats.scatter_add_(0, idx.reshape(-1, 1).expand(-1, C), weighted_grad.reshape(-1, C))
                    return grad_feats

            _fgk.triton = _TritonFallback()
            print("[INIT] flex_gemm Triton fallback patch applied.")
    except ImportError:
        print("[INIT] Could not patch flex_gemm triton fallback.")
    except Exception as e:
        print(f"[INIT] flex_gemm triton patch failed: {e}")


class TrellisState:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.pipeline = None

    def cleanup(self):
        """Clean up resources on shutdown."""
        self.pipeline = None
        torch.cuda.empty_cache()
        logger.info("Pipeline resources cleaned up")

    def initialize_pipeline(self, device=None):
        """Load the Trellis 2 pipeline and move it to the target device."""
        import os
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.65"
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

        _apply_patches()

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        print("[INIT] Loading Trellis 2 pipeline, please wait...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
        self.pipeline.to(device)
        print("[INIT] Trellis 2 pipeline ready.")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"VRAM allocated at startup: {allocated:.1f}MB")
            print(f"VRAM allocated at startup: {allocated:.1f}MB")


# Global state instance
state = TrellisState()