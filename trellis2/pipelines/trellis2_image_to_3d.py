# File: trellis2/pipelines/trellis2_image_to_3d.py
# trellis2/pipelines/trellis2_image_to_3d.py
import os
from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel

from trellis2.modules.sparse.conv import conv_flex_gemm
        


class _LazyCudaSubs:
    """Keeps subs on CPU; moves each to GPU on access, frees the previous one."""
    def __init__(self, subs_cpu, device):
        self._subs = subs_cpu  # list of SparseTensors on CPU
        self._device = device
        self._prev_idx = None

    def __getitem__(self, i):
        # Free the previous sub from GPU
        if self._prev_idx is not None and self._prev_idx != i:
            self._subs[self._prev_idx] = self._subs[self._prev_idx].to('cpu')
            torch.cuda.empty_cache()
        self._prev_idx = i
        self._subs[i] = self._subs[i].to(self._device)
        return self._subs[i]

    def __len__(self):
        return len(self._subs)


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @staticmethod
    def from_pretrained(path: str) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(Trellis2ImageTo3DPipeline, Trellis2ImageTo3DPipeline).from_pretrained(path)
        new_pipeline = Trellis2ImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        new_pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        new_pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        new_pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        new_pipeline.shape_slat_normalization = args['shape_slat_normalization']
        new_pipeline.tex_slat_normalization = args['tex_slat_normalization']

        new_pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        new_pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        new_pipeline.low_vram = args.get('low_vram', True)
        new_pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        new_pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        new_pipeline._device = 'cpu'

        return new_pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)
        self._convert_bf16_to_fp16_if_needed()


    def _convert_bf16_to_fp16_if_needed(self):
        """Convert all bfloat16 tensors to float16 for widest GPU compatibility."""
        import torch

        def _half_module(m):
            """Convert an nn.Module entirely to fp16."""
            for p in m.parameters():
                if p.dtype == torch.bfloat16:
                    p.data = p.data.half()
            for b in m.buffers():
                if b.dtype == torch.bfloat16:
                    b.data = b.data.half()
            # Catch unregistered tensor attrs (e.g. rope_phases)
            # and dtype attrs used by manual_cast()
            for sub in m.modules():
                for key, val in vars(sub).items():
                    if isinstance(val, torch.Tensor) and val.dtype == torch.bfloat16:
                        setattr(sub, key, val.half())
                    elif val is torch.bfloat16:
                        setattr(sub, key, torch.float16)

        # 1. Convert all pipeline models
        for name, model in self.models.items():
            _half_module(model)

        # 2. Convert any nn.Modules nested inside non-Module wrappers
        for extra in ('image_cond_model', 'rembg_model'):
            obj = getattr(self, extra, None)
            if obj is None:
                continue
            if isinstance(obj, torch.nn.Module):
                _half_module(obj)
                continue
            #else,  Walk one level into wrapper attrs to find nn.Modules
            for attr_val in vars(obj).values():
                if isinstance(attr_val, torch.nn.Module):
                    _half_module(attr_val)
                elif isinstance(attr_val, dict):
                    for v in attr_val.values():
                        if isinstance(v, torch.nn.Module):
                            _half_module(v)
                elif isinstance(attr_val, (list, tuple)):
                    for v in attr_val:
                        if isinstance(v, torch.nn.Module):
                            _half_module(v)
        

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        cond = self.image_cond_model(image)
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        import time
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso, device=self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        _t0 = time.time()
        if self.low_vram:
            flow_model.to(self.device)
        print(f"[TIMING] sparse_structure_flow_model.to(device): {time.time()-_t0:.2f}s")
        _t0 = time.time()
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        print(f"[TIMING] sparse_structure sampling (all steps): {time.time()-_t0:.2f}s")
        _t0 = time.time()
        if self.low_vram:
            flow_model.cpu()
        print(f"[TIMING] sparse_structure_flow_model.cpu(): {time.time()-_t0:.2f}s")
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        
        if self.low_vram:
            decoder.to(self.device)
            
        decoded = decoder(z_s)>0
        
        if self.low_vram:
            decoder.cpu()
            torch.cuda.empty_cache()

        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'], device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], device=slat.device)[None]
        slat = slat * std + mean
        
        return slat
    

    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # --- Phase 1: Low-Resolution Sampling ---
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        
        if self.low_vram:
            flow_model_lr.to(self.device)
            
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        
        if self.low_vram:
            flow_model_lr.cpu()
            torch.cuda.empty_cache()
            
        std = torch.tensor(self.shape_slat_normalization['std'], device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], device=slat.device)[None]
        slat = slat * std + mean
        
        # --- Phase 2: Upsampling ---
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            torch.cuda.empty_cache()

        # IMPORTANT: Drop the LR slat now. It is a large SparseTensor and we 
        # are about to generate an even larger one for the HR phase.
        del slat
        torch.cuda.empty_cache()
        
        # --- Phase 3: High-Resolution Coordinate Calculation ---
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution <= 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Cleanup temporary coordinate tensors to free VRAM for the HR Flow Model
        del hr_coords
        del quant_coords
        torch.cuda.empty_cache()
            
        # --- Phase 4: High-Resolution Sampling ---
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        
        if self.low_vram:
            flow_model.to(self.device)
            
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            torch.cuda.empty_cache()

        std = torch.tensor(self.shape_slat_normalization['std'], device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], device=slat.device)[None]
        slat = slat * std + mean
        
        return slat, hr_resolution


    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
        from o_voxel.convert import flexible_dual_grid_to_mesh
        from trellis2.representations import Mesh
        import torch.nn.functional as F

        decoder = self.models['shape_slat_decoder']
        decoder.set_resolution(resolution)

        # Step 1: Run decoder backbone only (parent class forward)
        decoded = SparseUnetVaeDecoder.forward(decoder, slat, return_subs=True)
        h, subs = decoded
        # Free accumulated neighbor maps from the decoder forward pass.
        # h and all subs share the same _spatial_cache dict via replace();
        # .clear() modifies it in-place so ALL references release GPU tensors.
        h._spatial_cache.clear()

        if os.environ.get('SPARSE_DEBUG') == '1':
            torch.cuda.synchronize()
            print(f"[VRAM] after shape decoder forward + cache clear: alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB  reserved={torch.cuda.memory_reserved()/1024**2:.0f}MB")

        # Step 2: Free decoder weights before mesh extraction
        if self.low_vram:
            decoder.cpu()
        torch.cuda.empty_cache()

        # Step 3: Extract mesh from decoded features
        # Cast to fp32 — the edge intersection test (> 0) and sigmoid vertex offsets
        # are precision-sensitive; fp16 rounding near zero creates stray quads
        # that appear as vertical line/hair artifacts in the mesh.
        h = h.replace(h.feats.float())
        voxel_margin = decoder.voxel_margin
        vertices = h.replace((1 + 2 * voxel_margin) * torch.sigmoid(h.feats[..., 0:3]) - voxel_margin)
        intersected = h.replace(h.feats[..., 3:6] > 0)
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
        mesh = [Mesh(*flexible_dual_grid_to_mesh(
            h.coords[:, 1:], v.feats, i.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=False
        )) for v, i, q in zip(vertices, intersected, quad_lerp)]

        del h, vertices, intersected, quad_lerp
        torch.cuda.empty_cache()

        return mesh, subs

    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'], device=shape_slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization['mean'], device=shape_slat.device)[None]
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'], device=slat.device)[None]
        mean = torch.tensor(self.tex_slat_normalization['mean'], device=slat.device)[None]
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            List[SparseTensor]: The decoded texture voxels
        """
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        return ret
    

    @torch.inference_mode()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.
        """
        from trellis2.modules.sparse.conv import conv_flex_gemm

        import os
        _vram_debug = os.environ.get('SPARSE_DEBUG') == '1'
        def _vram(label):
            if not _vram_debug:
                return
            torch.cuda.synchronize()
            a = torch.cuda.memory_allocated() / 1024**2
            r = torch.cuda.memory_reserved() / 1024**2
            print(f"[VRAM] {label}: alloc={a:.0f}MB  reserved={r:.0f}MB")

        _vram("decode_latent entry")

        # Offload tex_slat to CPU during shape decoding — not needed until texture phase
        tex_slat = tex_slat.to('cpu')
        torch.cuda.empty_cache()
        _vram("after tex_slat offload")

        # 1. Load shape decoder, run, and clear
        shape_dec = self.models['shape_slat_decoder']
        if shape_dec.dtype != torch.float16:
            shape_dec.convert_to_fp16()
            shape_dec.dtype = torch.float16
        if self.low_vram:
            shape_dec.low_vram = True
            shape_dec.from_latent.to(self.device)
            shape_dec.output_layer.to(self.device)
        _vram("after shape_dec to GPU")

        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        _vram("after decode_shape_slat")

        del shape_slat
        if self.low_vram:
            shape_dec.cpu()
        torch.cuda.empty_cache()
        _vram("after shape_dec offload + empty_cache")

        # Offload subs to CPU between decoder phases
        subs = [s.to('cpu') for s in subs]
        torch.cuda.empty_cache()
        # Offload meshes to CPU — not needed until final assembly
        for m in meshes:
            m.vertices = m.vertices.cpu()
            m.faces = m.faces.cpu()
        torch.cuda.empty_cache()
        _vram("after subs+meshes offload")

        # Reload tex_slat for texture decoding
        tex_slat = tex_slat.to(self.device)

        # 2. Load texture decoder, run, and clear
        tex_dec = self.models['tex_slat_decoder']
        if tex_dec.dtype != torch.float16:
            tex_dec.convert_to_fp16()
            tex_dec.dtype = torch.float16
        if self.low_vram:
            tex_dec.low_vram = True
            tex_dec.from_latent.to(self.device)
            tex_dec.output_layer.to(self.device)
        _vram("after tex_dec to GPU")

        # Wrap subs for lazy one-at-a-time GPU loading
        subs = _LazyCudaSubs(subs, self.device)

        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        # Free accumulated neighbor maps from the texture decoder forward pass.
        # Same in-place .clear() pattern — all references see the empty dict.
        tex_voxels._spatial_cache.clear()
        _vram("after decode_tex_slat + cache clear")

        del tex_slat
        del subs
        if self.low_vram:
            tex_dec.cpu()
        torch.cuda.empty_cache()
        _vram("after tex_dec offload + empty_cache")

        # 3. Assemble meshes
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.vertices = m.vertices.cuda()
            m.faces = m.faces.cuda()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:].contiguous(),  # .contiguous() breaks storage ref to full SparseTensor
                    attrs = v.feats.half(),    # Compress the final output to FP16 for the Rendering phase
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        _vram("after mesh assembly")
        return out_mesh


    @torch.inference_mode()
    def decode_and_cleanup(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode latents into meshes and offload decoders afterward.
        Separated from run() so cached latents can skip sampling
        and replay just the decode phase.
        """
        shape_slat._spatial_cache = {}
        tex_slat._spatial_cache = {}
        out_mesh = self.decode_latent(shape_slat, tex_slat, resolution)
        for name, model in self.models.items():
            if 'decoder' in name:
                model.cpu()
        torch.cuda.empty_cache()
        return out_mesh

    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        return_before_decode: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
            
        import time; _t0 = time.time()
        torch.manual_seed(seed)
        print(f"[TIMING] manual_seed: {time.time()-_t0:.2f}s")
        
        _t0 = time.time()
        if self.low_vram:
            self.image_cond_model.to(self.device)
        print(f"[TIMING] image_cond_model.to(device): {time.time()-_t0:.2f}s")
        
        _t0 = time.time()
        cond_512 = self._cast_cond(self.get_cond([image], 512))
        print(f"[TIMING] get_cond(512): {time.time()-_t0:.2f}s")
        
        cond_1024 = self._cast_cond(self.get_cond([image], 1024)) if pipeline_type != '512' else None
        
        if self.low_vram:
            self.image_cond_model.cpu()
            torch.cuda.empty_cache()

        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )

        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            del cond_512
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            del cond_512
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            del cond_1024
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            # cond_512 is no longer needed after shape cascade
            del cond_512
            shape_slat._spatial_cache = {}
            torch.cuda.empty_cache()

            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            # cond_1024 is no longer needed after texture sampling
            del cond_1024
            torch.cuda.empty_cache()
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            # cond_512 is no longer needed after shape cascade
            del cond_512
            shape_slat._spatial_cache = {}
            torch.cuda.empty_cache()

            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            # cond_1024 is no longer needed after texture sampling
            del cond_1024
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

        # Clear flow model neighbor caches from latent SparseTensors — no longer needed
        shape_slat._spatial_cache = {}
        tex_slat._spatial_cache = {}
        if return_before_decode:
            return shape_slat, tex_slat, res
        
        # Decode the generated latents into the final mesh
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        
        # Offload decoders after decode — no longer needed, frees VRAM for rendering
        for name, model in self.models.items():
            if 'decoder' in name:
                model.cpu()
        torch.cuda.empty_cache()
        
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh


    def _cast_cond(self, cond):
        """Cast any bf16 tensors in conditioning data to fp16."""
        import torch
        if cond is None:
            return None
        if isinstance(cond, torch.Tensor):
            return cond.half() if cond.dtype == torch.bfloat16 else cond
        if isinstance(cond, dict):
            return {k: self._cast_cond(v) for k, v in cond.items()}
        if isinstance(cond, (list, tuple)):
            return type(cond)(self._cast_cond(v) for v in cond)
        return cond