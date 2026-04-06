# File: trellis2/modules/sparse/conv/conv_flex_gemm.py
# trellis2/modules/sparse/conv/conv_flex_gemm.py
import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm 


LOW_VRAM_CONV = False


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    # overwrite the config setting to ensure we use the backend that doesn't need Triton
    flex_gemm.ops.spconv.set_algorithm(Algorithm.EXPLICIT_GEMM)
    flex_gemm.ops.spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)

    Co, Kd, Kh, Kw, Ci = self.weight.shape
    V = Kd * Kh * Kw
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    # Build neighbor cache if not available (uses CUDA kernels, no Triton needed)
    if neighbor_cache is None:
        from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
        neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(
            x.coords,
            torch.Size([*x.shape, *x.spatial_shape]),
            (Kw, Kh, Kd),
            self.dilation
        )
        # Only cache small neighbor maps — large ones (>256MB) are freed after each conv
        neighbor_map_bytes = neighbor_cache['neighbor_map'].numel() * neighbor_cache['neighbor_map'].element_size()
        if neighbor_map_bytes < 256 * 1024 * 1024:
            x.register_spatial_cache(neighbor_cache_key, neighbor_cache)

    # Chunked im2col + GEMM to cap peak VRAM
    feats = x.feats
    N = feats.shape[0]
    neighbor_map = neighbor_cache['neighbor_map']
    weight_mat = self.weight.reshape(Co, V * Ci).t()  # [V*Ci, Co]

    # Keep each chunk's im2col buffer under ~32MB for low-VRAM GPUs if flag is active (will be active only during some stages)
    im2col_bytes_per_voxel = V * Ci * feats.element_size()
    _cap = (32 * 1024 * 1024) if LOW_VRAM_CONV else (512 * 1024 * 1024)
    CHUNK = max(1024, _cap // im2col_bytes_per_voxel)

    output = torch.empty((N, Co), device=feats.device, dtype=feats.dtype)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk_map = neighbor_map[start:end]             # [chunk_n, V]
        chunk_n = end - start
        chunk_im2col = torch.zeros((chunk_n * V, Ci), device=feats.device, dtype=feats.dtype)
        flat_map = chunk_map.reshape(-1).int()
        mask = flat_map != -1
        chunk_im2col[mask] = feats[flat_map[mask].long()]
        chunk_im2col = chunk_im2col.view(chunk_n, V * Ci)
        if self.bias is not None:
            output[start:end] = torch.addmm(self.bias, chunk_im2col, weight_mat)
        else:
            output[start:end] = torch.mm(chunk_im2col, weight_mat)

    out = x.replace(output)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')
