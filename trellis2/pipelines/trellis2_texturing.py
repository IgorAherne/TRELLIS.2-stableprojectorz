# File: trellis2/pipelines/trellis2_texturing.py
# trellis2/pipelines/trellis2_texturing.py
"""
Shape-conditioned PBR texture generation (Microsoft TRELLIS.2, 903bfcf), adapted to this
fork:

- The UV bake rasterises with nvdiffrast, as upstream does. Where nvdiffrast does not
  import and o_voxel ships `uvraster` (a pure-PyTorch rasteriser with the same texel
  convention), it uses that instead.
- Low memory: `from_image_pipeline` shares the image-to-3D pipeline's texture flow models,
  texture decoder and DINOv3 instead of loading second copies; only the 0.7 GB shape
  encoder is new. The bf16 flow models run as fp16, as in the image pipeline.
- `retexture_glb` gives an existing GLB new base colour + metallic/roughness maps in its
  own UV layout, keeping its rig, animations and normal maps (trellis2/utils/glb_retexture).
  `run` is upstream's trimesh-in / trimesh-out API.

Important: the encoder's SparseTensor carries a spatial cache ('channel2spatial_*') that
the texture decoder uses to rebuild exactly the input surface's voxels. Never clear
`_spatial_cache` between encode_shape_slat and decode_tex_slat.
"""
import os
from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import trimesh
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from .. import models as _models_mod
import o_voxel
import cv2
import flex_gemm


def _rasterize_uv():
    """
    The bake's UV rasteriser, rasterize_uv(uvs, faces, size) -> (mask, tri, bary):
    uvs (V, 2) in [0, 1], faces (F, 3); per texel of a size x size map, whether a triangle
    covers it, which one (-1 if none) and the barycentric weights of its three corners.
    Texel (row r, col c) is sampled at uv ((c + 0.5) / size, (r + 0.5) / size); row 0 is
    v = 0. nvdiffrast as upstream, else o_voxel.uvraster where o_voxel has it.
    """
    try:
        import nvdiffrast.torch  # noqa: F401
        return _rasterize_uv_nvdiffrast
    except ImportError as e:
        try:
            from o_voxel.uvraster import rasterize_uv
        except ImportError:
            raise e
        return rasterize_uv


@torch.no_grad()
def _rasterize_uv_nvdiffrast(uvs: torch.Tensor, faces: torch.Tensor, size: int):
    import nvdiffrast.torch as dr
    device = uvs.device
    uvs = uvs.float().cuda()
    faces = faces.int().cuda().contiguous()
    pos = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=-1).unsqueeze(0)
    ctx = dr.RasterizeCudaContext()
    rast = torch.zeros((1, size, size, 4), device='cuda', dtype=torch.float32)
    for i in range(0, faces.shape[0], 100000):  # in chunks, as o_voxel's to_glb does
        chunk, _ = dr.rasterize(ctx, pos, faces[i:i + 100000], resolution=[size, size])
        hit = chunk[..., 3:4] > 0
        chunk[..., 3:4] += i  # triangle id + 1, offset by the chunk start
        rast = torch.where(hit, chunk, rast)
    del ctx
    rast = rast[0]
    mask = rast[..., 3] > 0
    tri = torch.where(mask, rast[..., 3].long() - 1, torch.full_like(rast[..., 3], -1, dtype=torch.long))
    u, v = rast[..., 0], rast[..., 1]  # nvdiffrast: attr = u * a0 + v * a1 + (1 - u - v) * a2
    bary = torch.stack([u, v, 1 - u - v], dim=-1) * mask.unsqueeze(-1)
    return mask.to(device), tri.to(device), bary.to(device)


class Trellis2TexturingPipeline(Pipeline):
    """
    Pipeline for texturing a given mesh from an image with TRELLIS.2.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'shape_slat_encoder',
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024'
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        tex_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
    ):
        if models is None:
            return
        super().__init__(models)
        self.tex_slat_sampler = tex_slat_sampler
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    # ------------------------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------------------------

    def _apply_config(self, args: dict) -> None:
        self.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        self.tex_slat_sampler_params = args['tex_slat_sampler']['params']
        self.shape_slat_normalization = args['shape_slat_normalization']
        self.tex_slat_normalization = args['tex_slat_normalization']
        self.low_vram = args.get('low_vram', True)
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2TexturingPipeline":
        """
        Load a pretrained model. Upstream's example passes config_file="texturing_pipeline.json".

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args
        pipeline._apply_config(args)
        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        return pipeline

    @classmethod
    def from_image_pipeline(
        cls,
        image_pipeline,
        path: str = "microsoft/TRELLIS.2-4B",
        config_file: str = "texturing_pipeline.json",
    ) -> "Trellis2TexturingPipeline":
        """
        Build a texturing pipeline that shares an already-loaded Trellis2ImageTo3DPipeline's
        models: the texture flow models, texture decoder, DINOv3 and background remover are
        the same objects, so the only new weights are the shape encoder's (~0.7 GB). A flow
        model the image pipeline didn't load stays absent here too, and a request that needs
        it fails with a clear error.
        """
        import json
        local = os.path.exists(f"{path}/{config_file}")
        if local:
            cfg_path = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(path, config_file)
        with open(cfg_path, 'r') as f:
            args = json.load(f)['args']

        shared = {}
        for k, v in args['models'].items():
            if k in image_pipeline.models:
                shared[k] = image_pipeline.models[k]
            elif k == 'shape_slat_encoder':
                try:
                    shared[k] = _models_mod.from_pretrained(f"{path}/{v}")
                except Exception:
                    shared[k] = _models_mod.from_pretrained(v)
            else:
                print(f"[TEXTURING] {k} not loaded (the image pipeline skipped it)")
        pipeline = cls(shared)
        pipeline._pretrained_args = args
        pipeline._apply_config(args)
        pipeline.low_vram = image_pipeline.low_vram
        pipeline.image_cond_model = image_pipeline.image_cond_model
        pipeline.rembg_model = image_pipeline.rembg_model
        pipeline.to(image_pipeline.device)
        return pipeline

    def to(self, device: torch.device) -> None:
        from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline as _Img
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)
        # Same treatment as the image pipeline: bf16 -> fp16 for older GPUs (a model the
        # image pipeline already converted is left alone).
        _Img._convert_bf16_to_fp16_if_needed(self)

    # ------------------------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------------------------

    @staticmethod
    def normalization(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
        """Centre and scale that put a Y-up mesh in Trellis' unit cube."""
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        center = (vmin + vmax) / 2
        scale = 0.99999 / (vmax - vmin).max()
        return center, scale

    @staticmethod
    def to_trellis_space(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
        """Y-up world points -> Trellis' Z-up unit cube ([-0.5, 0.5]^3)."""
        p = (points - center) * scale
        out = np.empty_like(p)
        out[:, 0] = p[:, 0]
        out[:, 1] = -p[:, 2]
        out[:, 2] = p[:, 1]
        return out

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocess the input mesh.
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        center, scale = self.normalization(vertices)
        vertices = self.to_trellis_space(vertices, center, scale)
        assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
        out = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        if hasattr(mesh, 'visual') and getattr(mesh.visual, 'uv', None) is not None:
            out.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv)
        return out

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
            if self.rembg_model is None:
                raise ValueError(
                    "Input image has no alpha channel and no background remover is loaded. "
                    "Send an RGBA image."
                )
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
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
            torch.cuda.empty_cache()
        if cond.dtype == torch.bfloat16:
            cond = cond.half()  # the flow models run in fp16 here
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    # ------------------------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------------------------

    def encode_shape_slat(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 1024,
    ) -> SparseTensor:
        """
        Encode the meshes to structured latent.

        Args:
            mesh (trimesh.Trimesh): The mesh to encode (already in Trellis space).
            resolution (int): The resolution of mesh

        Returns:
            SparseTensor: The encoded structured latent.
        """
        vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.faces)).long()

        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )

        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)

        encoder = self.models['shape_slat_encoder']
        if self.low_vram:
            encoder.to(self.device)
        shape_slat = encoder(vertices, intersected)
        if self.low_vram:
            encoder.cpu()
            torch.cuda.empty_cache()
        return shape_slat

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
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device, shape_slat.feats.dtype))
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
            torch.cuda.empty_cache()

        std = torch.tensor(self.tex_slat_normalization['std'], device=slat.device)[None]
        mean = torch.tensor(self.tex_slat_normalization['mean'], device=slat.device)[None]
        slat = slat * std + mean

        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
    ) -> SparseTensor:
        """
        Decode the structured latent. The decoder follows the encoder's subdivision (the
        spatial cache `slat` carries), so it rebuilds exactly the input surface's voxels.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        dec = self.models['tex_slat_decoder']
        if getattr(dec, 'dtype', None) != torch.float16 and hasattr(dec, 'convert_to_fp16'):
            dec.convert_to_fp16()
            dec.dtype = torch.float16
        if self.low_vram:
            # This fork's decoder streams its levels to the GPU one at a time when low_vram is on.
            dec.low_vram = True
            dec.from_latent.to(self.device)
            dec.output_layer.to(self.device)
        ret = dec(slat) * 0.5 + 0.5
        if self.low_vram:
            dec.cpu()
            torch.cuda.empty_cache()
        return ret

    def sample_voxels(self, pbr_voxel: SparseTensor, points: np.ndarray, resolution: int,
                      chunk: int = 2_000_000) -> np.ndarray:
        """
        Trilinear PBR attributes at Trellis-space points (N, 3) -> (N, C) floats 0..1.

        The voxels are a thin shell around the surface, so some of a surface point's eight
        trilinear neighbours are usually empty, and the sparse sampler counts an empty
        neighbour as zero: colours come out darker the more of them there are. Sampling a
        ones channel alongside gives the weight that landed on real voxels, and dividing by
        it keeps every texel at full brightness (normalised interpolation).
        """
        out = []
        C = pbr_voxel.feats.shape[1]
        feats = torch.cat([pbr_voxel.feats.float(), torch.ones_like(pbr_voxel.feats[:, :1], dtype=torch.float32)], dim=1)
        shape = torch.Size([pbr_voxel.shape[0], C + 1, *pbr_voxel.spatial_shape])
        for s in range(0, len(points), chunk):
            p = torch.from_numpy(np.ascontiguousarray(points[s:s + chunk])).float().to(self.device)
            v = flex_gemm.ops.grid_sample.grid_sample_3d(
                feats,
                pbr_voxel.coords,
                shape=shape,
                grid=((p + 0.5) * resolution).reshape(1, -1, 3),
                mode='trilinear',
            ).reshape(-1, C + 1).float()
            w = v[:, C:]
            v = torch.where(w > 1e-4, v[:, :C] / w.clamp_min(1e-4), v[:, :C])
            out.append(v.clamp(0, 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, C), dtype=np.float32)

    def texture_voxels(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        resolution: int = 1024,
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> SparseTensor:
        """
        The model part: Trellis-space mesh + image -> decoded PBR voxels (on the GPU).
        """
        say = progress or (lambda pct, msg: None)
        if resolution not in (512, 1024):
            raise ValueError("Texturing runs at 512 or 1024")
        key = f'tex_slat_flow_model_{resolution}'
        if key not in self.models:
            raise ValueError(f"No {resolution} texture flow model is loaded; use 512")
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        say(15, "Reading the image")
        cond = self.get_cond([image], resolution)
        say(25, "Encoding the shape")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        shape_slat = self.encode_shape_slat(mesh, resolution)
        say(35, "Sampling the texture")
        tex_slat = self.sample_tex_slat(cond, self.models[key], shape_slat, tex_slat_sampler_params)
        del cond, shape_slat
        torch.cuda.empty_cache()
        say(70, "Decoding the texture")
        pbr_voxel = self.decode_tex_slat(tex_slat)
        del tex_slat
        pbr_voxel._spatial_cache = {}  # done with the subdivision; let it go
        torch.cuda.empty_cache()
        return pbr_voxel

    # ------------------------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------------------------

    def postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        pbr_voxel: SparseTensor,
        resolution: int = 1024,
        texture_size: int = 1024,
    ) -> trimesh.Trimesh:
        """Upstream's output: the (Trellis-space) mesh with a fresh PBR material."""
        vertices = np.asarray(mesh.vertices).copy()
        faces = np.asarray(mesh.faces)
        normals = np.asarray(mesh.vertex_normals).copy()
        vertices_torch = torch.from_numpy(vertices).float().cuda()
        faces_torch = torch.from_numpy(faces).int().cuda()
        if hasattr(mesh, 'visual') and getattr(mesh.visual, 'uv', None) is not None:
            uvs = np.asarray(mesh.visual.uv).copy()
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs_torch = torch.from_numpy(uvs).float().cuda()
        else:
            import cumesh
            _cumesh = cumesh.CuMesh()
            _cumesh.init(vertices_torch, faces_torch)
            vertices_torch, faces_torch, uvs_torch, vmap = _cumesh.uv_unwrap(return_vmaps=True)
            vertices_torch = vertices_torch.cuda()
            faces_torch = faces_torch.cuda()
            uvs_torch = uvs_torch.cuda()
            vertices = vertices_torch.cpu().numpy()
            faces = faces_torch.cpu().numpy()
            uvs = uvs_torch.cpu().numpy()
            normals = normals[vmap.cpu().numpy()]

        # rasterize
        mask, tri_id, bary = _rasterize_uv()(uvs_torch, faces_torch, texture_size)
        tri = vertices_torch[faces_torch.long()[tri_id[mask]]]
        pos = (tri * bary[mask].unsqueeze(-1)).sum(dim=1)

        attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=self.device)
        attrs[mask] = torch.from_numpy(self.sample_voxels(pbr_voxel, pos.cpu().numpy(), resolution)).to(self.device)

        # construct mesh
        mask = mask.cpu().numpy()
        base_color = np.clip(attrs[..., self.pbr_attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., self.pbr_attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., self.pbr_attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., self.pbr_attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # extend
        mask = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask, 1, cv2.INPAINT_TELEA)[..., None]

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=True,
        )

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
        normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]
        uvs[:, 1] = 1 - uvs[:, 1]  # Flip UV V-coordinate

        textured_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
        )

        return textured_mesh

    @torch.no_grad()
    def run(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        resolution: int = 1024,
        texture_size: int = 2048,
    ) -> trimesh.Trimesh:
        """
        Run the pipeline (upstream API).

        Args:
            mesh (trimesh.Trimesh): The mesh to texture.
            image (Image.Image): The image prompt.
            seed (int): The random seed.
            tex_slat_sampler_params (dict): Additional parameters for the texture latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        mesh = self.preprocess_mesh(mesh)
        pbr_voxel = self.texture_voxels(
            np.asarray(mesh.vertices), np.asarray(mesh.faces), image, seed,
            tex_slat_sampler_params, preprocess_image, resolution)
        return self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size)

    @torch.no_grad()
    def retexture_glb(
        self,
        glb: bytes,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        resolution: int = 1024,
        texture_size: int = 0,
        max_texture_size: int = 4096,
        uv_mode: str = 'auto',
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[bytes, dict]:
        """
        Give a GLB new PBR textures in its own UV layout; everything else in the file is
        kept (rig, clips, normal maps, node tree).

        texture_size: 0 = each material keeps its current base colour size.
        uv_mode: 'keep' bakes into the existing UVs; 'new' unwraps afresh (cumesh), splitting
          vertices but carrying skin weights and every other attribute across; 'auto' keeps
          a usable layout and unwraps one that is not (stacked palette UVs, no UVs).
        Returns the new GLB bytes and a report.
        """
        from ..utils import glb_retexture as G
        say = progress or (lambda pct, msg: None)
        say(5, "Reading the model")
        gltf, binary = G.read_glb(glb)
        surfs = G.surfaces(gltf, binary)
        verts, faces = G.combined(surfs)
        center, scale = self.normalization(verts)

        by_mat: Dict[int, list] = {}
        for s in surfs:
            if s.material is not None:
                by_mat.setdefault(s.material, []).append(s)
        if not by_mat:
            raise ValueError("The model has no materials to texture")

        # Decide each material's UVs before the GPU work, so a bad request fails fast.
        report = {'materials': [], 'resolution': resolution, 'seed': seed}
        plan = {}
        for m, ss in by_mat.items():
            if all(s.uvs is not None for s in ss):
                offsets = np.cumsum([0] + [len(s.positions) for s in ss[:-1]])
                health = G.uv_health(np.concatenate([s.uvs for s in ss]),
                                     np.concatenate([s.faces + o for s, o in zip(ss, offsets)]),
                                     _rasterize_uv())
            else:
                health = {'usable': False, 'reason': 'no UVs'}
            mode = uv_mode
            cutout = gltf['materials'][m].get('alphaMode', 'OPAQUE') != 'OPAQUE'
            if mode == 'auto':
                # A cut-out or blended material (feather cards, hair, foliage) gets its shape
                # from its own alpha, laid out in its own UVs, which often stack cards on one
                # another by design: keep them, and keep that alpha.
                mode = 'keep' if (health['usable'] or (cutout and health.get('reason') != 'no UVs')) else 'new'
            if mode == 'keep' and health.get('reason') == 'no UVs':
                raise ValueError(f"Material {m} has primitives without UVs; use uv_mode=new")
            plan[m] = mode
            report['materials'].append({'material': m, 'name': gltf['materials'][m].get('name'), 'uvs': mode,
                                        'uvHealth': health, 'primitives': len(ss)})

        # The model: encode the whole surface, sample, decode.
        tverts = self.to_trellis_space(verts, center, scale)
        assert np.all(tverts >= -0.5) and np.all(tverts <= 0.5), 'vertices out of range'
        pbr_voxel = self.texture_voxels(tverts, faces, image, seed, tex_slat_sampler_params,
                                        preprocess_image, resolution, progress)

        def sample(points_world: np.ndarray) -> np.ndarray:
            return self.sample_voxels(pbr_voxel, self.to_trellis_space(points_world, center, scale), resolution)

        editor = G.GlbEditor(gltf, binary)
        rasterize_uv = _rasterize_uv()
        bake_device = str(self.device) if str(self.device).startswith('cuda') else 'cuda'
        sampler_idx = None
        for i, (m, ss) in enumerate(sorted(by_mat.items())):
            say(80 + int(15 * i / len(by_mat)), f"Baking material {i + 1} of {len(by_mat)}")
            size = texture_size or G.material_texture_size(gltf, binary, m, 2048)
            size = int(max(256, min(size, max_texture_size)))
            if plan[m] == 'new':
                import cumesh
                for s in ss:
                    cm = cumesh.CuMesh()
                    cm.init(torch.from_numpy(s.positions).float().cuda(), torch.from_numpy(s.faces).int().cuda())
                    _v, nf, nuv, vmap = cm.uv_unwrap(return_vmaps=True)
                    vmap = vmap.long().cpu().numpy()
                    nf = nf.long().cpu().numpy()
                    nuv = nuv.float().cpu().numpy().astype(np.float64)
                    G.replace_primitive_geometry(editor, binary, s, vmap, nf, nuv)
                    s.positions, s.faces, s.uvs = s.positions[vmap], nf, nuv
            baked = G.bake_material(ss, sample, size, rasterize_uv, device=bake_device)
            lay = self.pbr_attr_layout
            base = np.clip(baked[..., lay['base_color']] * 255 + 0.5, 0, 255).astype(np.uint8)
            metal = np.clip(baked[..., lay['metallic']] * 255 + 0.5, 0, 255).astype(np.uint8)
            rough = np.clip(baked[..., lay['roughness']] * 255 + 0.5, 0, 255).astype(np.uint8)
            mat = gltf['materials'][m]
            mime = G.material_mime(gltf, m)
            keep_alpha = G.original_alpha(gltf, binary, m, size) if plan[m] == 'keep' else None
            if mat.get('alphaMode', 'OPAQUE') != 'OPAQUE':
                a = keep_alpha if keep_alpha is not None else np.clip(baked[..., lay['alpha']] * 255 + 0.5, 0, 255).astype(np.uint8)[..., 0]
                base_img = Image.fromarray(np.concatenate([base, a[..., None]], axis=-1), 'RGBA')
                base_mime = 'image/png'
            else:
                base_img = Image.fromarray(base, 'RGB')
                base_mime = mime
            # glTF reads roughness from G and metalness from B; R is left white so an
            # occlusion lookup that shares the map would find "no occlusion".
            mr_img = Image.fromarray(np.concatenate([np.full_like(metal, 255), rough, metal], axis=-1), 'RGB')
            if sampler_idx is None:
                samplers_ = gltf.setdefault('samplers', [])
                samplers_.append({'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497})
                sampler_idx = len(samplers_) - 1
            name = mat.get('name', f'material{m}')
            bt = editor.add_texture(editor.add_image(base_img, base_mime, f'{name}_basecolor'), sampler_idx)
            mt = editor.add_texture(editor.add_image(mr_img, mime, f'{name}_metallic_roughness'), sampler_idx)
            pbr = mat.setdefault('pbrMetallicRoughness', {})
            tc = ss[0].texcoord
            pbr['baseColorTexture'] = {'index': bt, **({'texCoord': tc} if tc else {})}
            pbr['metallicRoughnessTexture'] = {'index': mt, **({'texCoord': tc} if tc else {})}
            factor = pbr.get('baseColorFactor', [1, 1, 1, 1])
            pbr['baseColorFactor'] = [1.0, 1.0, 1.0, float(factor[3]) if len(factor) > 3 else 1.0]
            pbr['metallicFactor'] = 1.0
            pbr['roughnessFactor'] = 1.0
            cov = float(baked[..., -1].mean())
            for r in report['materials']:
                if r['material'] == m:
                    r.update({'textureSize': size, 'coverage': round(cov, 3),
                              'metallicMean': round(float(metal.mean()) / 255, 3),
                              'roughnessMean': round(float(rough.mean()) / 255, 3)})
        del pbr_voxel
        torch.cuda.empty_cache()
        say(97, "Writing the GLB")
        return editor.write(), report
