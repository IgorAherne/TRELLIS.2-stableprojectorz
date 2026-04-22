# File: trellis2/modules/sparse/conv/conv_flex_gemm.py
# trellis2/modules/sparse/conv/conv_flex_gemm.py
import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm 


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
    N = x.feats.shape[0]
    _nb = N * V * 4
    _ob = N * Co * x.feats.element_size()
    torch.cuda.synchronize()
    print(f"[CONV] N={N} Ci={Ci} Co={Co} neighbor_map={_nb/1024/1024:.0f}MB offload_out={_ob >= 256*1024*1024} feats_bytes={x.feats.numel()*x.feats.element_size()/1024/1024:.0f}MB")
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    # Build neighbor cache if not available (uses CUDA kernels, no Triton needed)
    if neighbor_cache is None:
        from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
        neighbor_map_bytes = x.feats.shape[0] * V * 4  # uint32
        if neighbor_map_bytes >= 256 * 1024 * 1024:
            # Large map � bounce feats to CPU so feats (1.37GB) and
            # neighbor_map (1.16GB) never coexist on GPU during build.
            feats_cpu = x.feats.cpu()
            x.feats = torch.empty(0, dtype=x.feats.dtype, device='cpu')
            torch.cuda.empty_cache()
            neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(
                x.coords,
                torch.Size([*x.shape, *x.spatial_shape]),
                (Kw, Kh, Kd),
                self.dilation
            )
            neighbor_map = neighbor_cache['neighbor_map'].cpu()
            del neighbor_cache
            torch.cuda.empty_cache()
            x.feats = feats_cpu.to(x.coords.device)
            del feats_cpu
            stream_from_cpu = True
        else:
            neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(
                x.coords,
                torch.Size([*x.shape, *x.spatial_shape]),
                (Kw, Kh, Kd),
                self.dilation
            )
            x.register_spatial_cache(neighbor_cache_key, neighbor_cache)
            neighbor_map = neighbor_cache['neighbor_map']
            stream_from_cpu = False
    else:
        neighbor_map = neighbor_cache['neighbor_map']
        stream_from_cpu = False

    # Chunked im2col + GEMM to cap peak VRAM
    feats = x.feats
    N = feats.shape[0]
    weight_mat = self.weight.reshape(Co, V * Ci).t()  # [V*Ci, Co]

    # Keep each chunk's im2col buffer under ~64MB for low-VRAM GPUs
    im2col_bytes_per_voxel = V * Ci * feats.element_size()
    _cap = (64 * 1024 * 1024)
    CHUNK = max(1024, _cap // im2col_bytes_per_voxel)

    # For large outputs, accumulate to CPU so feats + output
    # never coexist on GPU.  At 10.7M voxels � 64ch � fp16 each is 1.37GB;
    # overlapping them pushes peak to 3GB.  Writing to CPU keeps peak at ~1.5GB.
    output_bytes = N * Co * feats.element_size()
    offload_output = output_bytes >= 256 * 1024 * 1024

    if offload_output:
        output = torch.empty((N, Co), dtype=feats.dtype)
    else:
        output = torch.empty((N, Co), device=feats.device, dtype=feats.dtype)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk_map = neighbor_map[start:end]             # [chunk_n, V]
        if stream_from_cpu:
            chunk_map = chunk_map.to(feats.device, non_blocking=True)
        chunk_n = end - start
        chunk_im2col = torch.zeros((chunk_n * V, Ci), device=feats.device, dtype=feats.dtype)
        flat_map = chunk_map.reshape(-1).int()
        mask = flat_map != -1
        chunk_im2col[mask] = feats[flat_map[mask].long()]
        chunk_im2col = chunk_im2col.view(chunk_n, V * Ci)
        if self.bias is not None:
            chunk_result = torch.addmm(self.bias, chunk_im2col, weight_mat)
        else:
            chunk_result = torch.mm(chunk_im2col, weight_mat)
        if offload_output:
            output[start:end] = chunk_result.cpu()
            del chunk_result
        else:
            output[start:end] = chunk_result

    if offload_output:
        # Free input feats from GPU, then reload output from CPU
        device = feats.device
        x.feats = torch.empty(0, dtype=feats.dtype, device='cpu')
        del feats
        torch.cuda.empty_cache()
        output = output.to(device)

    out = x.replace(output)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')
