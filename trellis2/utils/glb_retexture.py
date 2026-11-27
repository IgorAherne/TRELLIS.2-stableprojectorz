# File: trellis2/utils/glb_retexture.py
# trellis2/utils/glb_retexture.py
"""
Give an existing GLB new PBR textures from Trellis2TexturingPipeline, keeping everything
else about the file: its meshes, UVs, skins, animations, normal maps and node tree.

The upstream texturing pipeline loads a mesh with trimesh and exports a brand-new static
mesh. That loses a rig and its clips, splits vertices, and replaces UVs someone may have
laid out by hand. This module works on the glTF itself instead:

  1. `read_glb` + `surfaces` read every textured primitive's triangles in world space at
     its bind pose (skinned meshes are skinned with their own inverse bind matrices).
  2. The caller encodes all of them together as one shape and samples PBR voxels.
  3. `bake_material` rasterises each material's own UV layout and samples the voxels at
     the surface point under every texel: base colour (+ alpha) and metallic-roughness.
  4. `write_glb` points each material at its new images and drops whatever nothing
     references any more.

A primitive whose UVs cannot hold a texture (overlapping palette swatches, as on Synty
characters, or no UVs at all) can be unwrapped afresh (`uv_mode='new'`): its vertices
are split where the new seams need it and every vertex attribute (skin weights, normals,
colours) is carried across with the split, so a rig survives that too.

Zero third-party dependencies beyond numpy and PIL; no Draco or meshopt compression.
"""
import io
import json
import struct
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


# ----------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------

def read_glb(data: bytes) -> Tuple[dict, bytearray]:
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError('not a glTF 2.0 binary (.glb)')
    off, gltf, binary = 12, None, bytearray()
    while off < length:
        clen, ctype = struct.unpack_from('<II', data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == CHUNK_JSON:
            gltf = json.loads(bytes(chunk).decode('utf-8'))
        elif ctype == CHUNK_BIN:
            binary = bytearray(chunk)
        off += 8 + clen
    if gltf is None:
        raise ValueError('GLB has no JSON chunk')
    for ext in gltf.get('extensionsRequired', []):
        raise ValueError(f'GLB requires {ext}, which this retexturer cannot read (decompress it first)')
    for buf in gltf.get('buffers', [])[1:] + [b for b in gltf.get('buffers', [])[:1] if 'uri' in b]:
        raise ValueError('GLB references external buffers; only self-contained .glb files are supported')
    return gltf, binary


def read_accessor(gltf: dict, binary: bytearray, index: int) -> np.ndarray:
    acc = gltf['accessors'][index]
    dtype = np.dtype(COMPONENT[acc['componentType']])
    n = NCOMP[acc['type']]
    count = acc['count']
    if 'bufferView' in acc:
        view = gltf['bufferViews'][acc['bufferView']]
        start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        stride = view.get('byteStride', 0) or dtype.itemsize * n
        if stride == dtype.itemsize * n:
            out = np.frombuffer(binary, dtype=dtype, count=count * n, offset=start).reshape(count, n).copy()
        else:
            raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * n, offset=start)
            rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * n), strides=(stride, 1))
            out = rows.copy().view(dtype).reshape(count, n)
    else:
        out = np.zeros((count, n), dtype=dtype)
    if 'sparse' in acc:
        sp = acc['sparse']
        idx_view = gltf['bufferViews'][sp['indices']['bufferView']]
        idx = np.frombuffer(binary, dtype=COMPONENT[sp['indices']['componentType']], count=sp['count'],
                            offset=idx_view.get('byteOffset', 0) + sp['indices'].get('byteOffset', 0))
        val_view = gltf['bufferViews'][sp['values']['bufferView']]
        val = np.frombuffer(binary, dtype=dtype, count=sp['count'] * n,
                            offset=val_view.get('byteOffset', 0) + sp['values'].get('byteOffset', 0)).reshape(-1, n)
        out[idx.astype(np.int64)] = val
    if acc.get('normalized') and dtype.kind in 'iu':
        out = out.astype(np.float32) / np.iinfo(dtype).max
    return out


def _node_local(node: dict) -> np.ndarray:
    if 'matrix' in node:
        return np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get('translation', [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    s = np.array(node.get('scale', [1, 1, 1]), dtype=np.float64)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


def world_matrices(gltf: dict) -> Dict[int, np.ndarray]:
    nodes = gltf.get('nodes', [])
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i
    cache: Dict[int, np.ndarray] = {}

    def world(i):
        if i not in cache:
            local = _node_local(nodes[i])
            cache[i] = world(parent[i]) @ local if i in parent else local
        return cache[i]

    for i in range(len(nodes)):
        world(i)
    return cache


class Surface:
    """One primitive's triangles, in world space at its bind pose."""
    def __init__(self, mesh: int, prim: int, material: Optional[int], positions: np.ndarray,
                 faces: np.ndarray, uvs: Optional[np.ndarray], texcoord: int):
        self.mesh, self.prim, self.material = mesh, prim, material
        self.positions, self.faces, self.uvs, self.texcoord = positions, faces, uvs, texcoord


def _texcoord_of(gltf: dict, material: Optional[int]) -> int:
    if material is None:
        return 0
    pbr = gltf['materials'][material].get('pbrMetallicRoughness', {})
    return pbr.get('baseColorTexture', {}).get('texCoord', 0)


def surfaces(gltf: dict, binary: bytearray) -> List[Surface]:
    """
    Every triangle primitive that is drawn, with world-space positions. A mesh drawn by
    several nodes is taken once (its first instance): textures live in UV space, so one
    copy is all a bake needs. Skinned meshes are posed with their skin's inverse bind
    matrices and the joints' rest transforms, which is how a viewer draws them unanimated.
    """
    worlds = world_matrices(gltf)
    out: List[Surface] = []
    seen = set()
    for ni, node in enumerate(gltf.get('nodes', [])):
        mi = node.get('mesh')
        if mi is None or mi in seen:
            continue
        seen.add(mi)
        skin = gltf['skins'][node['skin']] if 'skin' in node else None
        if skin is not None:
            joints = skin['joints']
            if 'inverseBindMatrices' in skin:
                ibm = read_accessor(gltf, binary, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
            else:
                ibm = np.repeat(np.eye(4)[None], len(joints), axis=0)
            joint_mats = np.stack([worlds[j] for j in joints]) @ ibm
        for pi, prim in enumerate(gltf['meshes'][mi]['primitives']):
            if prim.get('mode', 4) != 4:
                continue  # points / lines: nothing to texture
            attrs = prim['attributes']
            pos = read_accessor(gltf, binary, attrs['POSITION']).astype(np.float64)
            hom = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
            if skin is not None and 'JOINTS_0' in attrs and 'WEIGHTS_0' in attrs:
                j = read_accessor(gltf, binary, attrs['JOINTS_0']).astype(np.int64)
                w = read_accessor(gltf, binary, attrs['WEIGHTS_0']).astype(np.float64)
                ws = w.sum(axis=1, keepdims=True)
                w = np.where(ws > 0, w / np.maximum(ws, 1e-12), np.array([[1.0, 0, 0, 0]]))
                skinned = np.einsum('vk,vkij,vj->vi', w, joint_mats[j], hom)
                world = skinned[:, :3]
            else:
                world = (hom @ worlds[ni].T)[:, :3]
            if 'indices' in prim:
                faces = read_accessor(gltf, binary, prim['indices']).reshape(-1, 3).astype(np.int64)
            else:
                faces = np.arange(len(pos), dtype=np.int64).reshape(-1, 3)
            material = prim.get('material')
            tc = _texcoord_of(gltf, material)
            uv_key = f'TEXCOORD_{tc}'
            uvs = read_accessor(gltf, binary, attrs[uv_key]).astype(np.float64) if uv_key in attrs else None
            out.append(Surface(mi, pi, material, world, faces, uvs, tc))
    if not out:
        raise ValueError('GLB has no triangle meshes')
    return out


def combined(surfs: List[Surface]) -> Tuple[np.ndarray, np.ndarray]:
    """All surfaces as one vertex/face soup (for encoding the shape)."""
    verts, faces, base = [], [], 0
    for s in surfs:
        verts.append(s.positions)
        faces.append(s.faces + base)
        base += len(s.positions)
    return np.concatenate(verts), np.concatenate(faces)


# ----------------------------------------------------------------------------------------
# UV checks
# ----------------------------------------------------------------------------------------

def uv_health(uvs: Optional[np.ndarray], faces: np.ndarray, rasterize_uv: Callable,
              size: int = 512) -> dict:
    """
    Can this UV layout hold a baked texture? Palette-swatch UVs (Synty) put many triangles
    on the same few texels, and a bake into them gives every triangle the colour of
    whichever one was drawn last. Measured as how much UV area is stacked on itself.
    `rasterize_uv` is the bake's own rasteriser (trellis2_texturing._rasterize_uv); it runs
    on the GPU when there is one.
    """
    if uvs is None:
        return {'usable': False, 'reason': 'no UVs'}
    t = uvs[faces]
    area = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    degenerate = float((area < 1e-9).mean())
    total = float(area.sum())
    # Covered area: the share of texels whose centre falls inside any triangle.
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask, _, _ = rasterize_uv(torch.from_numpy(np.clip(uvs, 0.0, 1.0)).float().to(device),
                              torch.from_numpy(np.ascontiguousarray(faces)).long().to(device), size)
    covered = float(mask.float().mean())
    del mask
    stacking = total / max(covered, 1e-9)
    # Under 15% of the texture in use (a projected or hand-packed layout with most of the
    # atlas empty) means a 2048 map is really a ~780 px one: a fresh unwrap uses ~60%.
    usable = degenerate < 0.2 and stacking < 1.5 and covered >= 0.15
    reason = None if usable else (
        f'{degenerate * 100:.0f}% of triangles have no UV area' if degenerate >= 0.2 else
        f'UV islands overlap ({stacking:.1f}x stacked)' if stacking >= 1.5 else
        f'UVs use only {covered * 100:.1f}% of the texture')
    return {'usable': usable, 'reason': reason, 'stacking': round(stacking, 2),
            'covered': round(covered, 3), 'degenerate': round(degenerate, 3)}


# ----------------------------------------------------------------------------------------
# Baking
# ----------------------------------------------------------------------------------------

def bake_material(
    surfs: List[Surface],
    sample: Callable[[np.ndarray], np.ndarray],
    size: int,
    rasterize_uv: Callable,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Rasterise the surfaces' shared UV layout at `size` x `size` and sample PBR attributes
    at the world point under each covered texel. `sample(points_world) -> (N, C)` floats in
    0..1 (base colour 0:3, metallic 3, roughness 4, alpha 5). Returns (size, size, C) float32
    with uncovered texels filled from their neighbours (so mip maps and bilinear filtering
    never pull in black), plus the coverage mask as a final boolean channel.
    """
    import torch
    import cv2
    uv_all, faces_all, pos_all, base = [], [], [], 0
    for s in surfs:
        uv_all.append(s.uvs)
        faces_all.append(s.faces + base)
        pos_all.append(s.positions)
        base += len(s.positions)
    uvs = torch.from_numpy(np.concatenate(uv_all)).float().to(device)
    faces = torch.from_numpy(np.concatenate(faces_all)).long().to(device)
    pos = torch.from_numpy(np.concatenate(pos_all)).double().to(device)
    # glTF UV (0,0) is the image's top-left corner, and the rasteriser's row 0 is v = 0, so the
    # raster's rows are the image's rows as they will be written.
    mask, tri_id, bary = rasterize_uv(uvs, faces.int(), size)
    tri = pos[faces[tri_id[mask]]]                                   # (N, 3, 3)
    points = (tri * bary[mask].double().unsqueeze(-1)).sum(dim=1)   # (N, 3)
    del tri, tri_id, bary
    attrs_cov = sample(points.cpu().numpy())                         # (N, C)
    mask_np = mask.cpu().numpy()
    C = attrs_cov.shape[1]
    img = np.zeros((size, size, C), dtype=np.float32)
    img[mask_np] = attrs_cov
    # Fill the gutters: cv2.inpaint on 8-bit channels, as o_voxel's to_glb does.
    hole = (~mask_np).astype(np.uint8)
    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    filled = np.empty_like(u8)
    for c0 in range(0, C, 3):
        part = u8[..., c0:c0 + 3]
        if part.shape[-1] == 3:
            filled[..., c0:c0 + 3] = cv2.inpaint(np.ascontiguousarray(part), hole, 3, cv2.INPAINT_TELEA)
        else:
            for k in range(part.shape[-1]):
                filled[..., c0 + k] = cv2.inpaint(np.ascontiguousarray(part[..., k]), hole, 3, cv2.INPAINT_TELEA)
    out = filled.astype(np.float32) / 255.0
    out[mask_np] = img[mask_np]  # covered texels keep full precision
    return np.concatenate([out, mask_np[..., None].astype(np.float32)], axis=-1)


# ----------------------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------------------

def _encode(img: Image.Image, mime: str) -> bytes:
    buf = io.BytesIO()
    if mime == 'image/jpeg':
        img.convert('RGB').save(buf, format='JPEG', quality=92, optimize=True)
    else:
        img.save(buf, format='PNG', optimize=False, compress_level=6)
    return buf.getvalue()


def _image_size(gltf: dict, binary: bytearray, image_index: int) -> Optional[Tuple[int, int]]:
    img = gltf['images'][image_index]
    if 'bufferView' not in img:
        return None
    v = gltf['bufferViews'][img['bufferView']]
    raw = bytes(binary[v.get('byteOffset', 0): v.get('byteOffset', 0) + v['byteLength']])
    try:
        return Image.open(io.BytesIO(raw)).size
    except Exception:
        return None


def _image_pil(gltf: dict, binary: bytearray, image_index: int) -> Optional[Image.Image]:
    img = gltf['images'][image_index]
    if 'bufferView' not in img:
        return None
    v = gltf['bufferViews'][img['bufferView']]
    raw = bytes(binary[v.get('byteOffset', 0): v.get('byteOffset', 0) + v['byteLength']])
    try:
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def material_texture_size(gltf: dict, binary: bytearray, material: int, default: int) -> int:
    """The size the material's current base colour texture has (its larger side)."""
    tex = gltf['materials'][material].get('pbrMetallicRoughness', {}).get('baseColorTexture')
    if tex is None:
        return default
    src = gltf['textures'][tex['index']].get('source')
    wh = _image_size(gltf, binary, src) if src is not None else None
    return max(wh) if wh else default


def material_mime(gltf: dict, material: int) -> str:
    tex = gltf['materials'][material].get('pbrMetallicRoughness', {}).get('baseColorTexture')
    if tex is None:
        return 'image/png'
    src = gltf['textures'][tex['index']].get('source')
    return gltf['images'][src].get('mimeType', 'image/png') if src is not None else 'image/png'


def original_alpha(gltf: dict, binary: bytearray, material: int, size: int) -> Optional[np.ndarray]:
    """A cut-out or blended material's own alpha channel, resampled to `size` (or None)."""
    mat = gltf['materials'][material]
    if mat.get('alphaMode', 'OPAQUE') == 'OPAQUE':
        return None
    tex = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture')
    if tex is None:
        return None
    src = gltf['textures'][tex['index']].get('source')
    pil = _image_pil(gltf, binary, src) if src is not None else None
    if pil is None or 'A' not in pil.getbands():
        return None
    a = pil.getchannel('A').resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(a, dtype=np.uint8)


class GlbEditor:
    """Collects new images / accessors and writes a compacted GLB."""

    def __init__(self, gltf: dict, binary: bytearray):
        self.gltf = gltf
        self.binary = binary
        self.extra: List[bytes] = []  # appended bufferView payloads, in order

    def _add_view(self, payload: bytes, target: Optional[int] = None) -> int:
        views = self.gltf.setdefault('bufferViews', [])
        view = {'buffer': 0, 'byteOffset': -1 - len(self.extra), 'byteLength': len(payload)}
        if target is not None:
            view['target'] = target
        self.extra.append(payload)
        views.append(view)
        return len(views) - 1

    def add_image(self, img: Image.Image, mime: str, name: str) -> int:
        view = self._add_view(_encode(img, mime))
        images = self.gltf.setdefault('images', [])
        images.append({'bufferView': view, 'mimeType': mime, 'name': name})
        return len(images) - 1

    def add_texture(self, image: int, sampler: Optional[int]) -> int:
        textures = self.gltf.setdefault('textures', [])
        t = {'source': image}
        if sampler is not None:
            t['sampler'] = sampler
        textures.append(t)
        return len(textures) - 1

    def add_accessor(self, arr: np.ndarray, gl_type: str, component: int, target: Optional[int],
                     normalized: bool = False, minmax: bool = False) -> int:
        arr = np.ascontiguousarray(arr.astype(COMPONENT[component]))
        view = self._add_view(arr.tobytes(), target)
        acc = {'bufferView': view, 'componentType': component, 'count': int(arr.shape[0]), 'type': gl_type}
        if normalized:
            acc['normalized'] = True
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc['min'] = flat.min(axis=0).tolist()
            acc['max'] = flat.max(axis=0).tolist()
        accs = self.gltf.setdefault('accessors', [])
        accs.append(acc)
        return len(accs) - 1

    # -- compaction ---------------------------------------------------------------------

    def _prune(self):
        """Drop accessors, images, textures and bufferViews that nothing references."""
        g = self.gltf
        used_tex = set()

        def walk_tex(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k.endswith('Texture') and isinstance(v, dict) and 'index' in v:
                        used_tex.add(v['index'])
                    walk_tex(v)
            elif isinstance(o, list):
                for v in o:
                    walk_tex(v)
        walk_tex(g.get('materials', []))
        used_img = {g['textures'][t]['source'] for t in used_tex if 'source' in g['textures'][t]}
        for t in used_tex:  # images named by texture extensions (webp/basisu)
            for ext in g['textures'][t].get('extensions', {}).values():
                if 'source' in ext:
                    used_img.add(ext['source'])
        used_acc = set()
        for m in g.get('meshes', []):
            for p in m['primitives']:
                used_acc.update(p['attributes'].values())
                if 'indices' in p:
                    used_acc.add(p['indices'])
                for t in p.get('targets', []):
                    used_acc.update(t.values())
        for s in g.get('skins', []):
            if 'inverseBindMatrices' in s:
                used_acc.add(s['inverseBindMatrices'])
        for a in g.get('animations', []):
            for smp in a['samplers']:
                used_acc.update([smp['input'], smp['output']])

        def remap(n, used):
            order = sorted(used)
            return order, {old: new for new, old in enumerate(order)}

        acc_order, acc_map = remap(len(g.get('accessors', [])), used_acc)
        img_order, img_map = remap(len(g.get('images', [])), used_img)
        tex_order, tex_map = remap(len(g.get('textures', [])), used_tex)

        used_view = set()
        for i in acc_order:
            a = g['accessors'][i]
            if 'bufferView' in a:
                used_view.add(a['bufferView'])
            if 'sparse' in a:
                used_view.update([a['sparse']['indices']['bufferView'], a['sparse']['values']['bufferView']])
        for i in img_order:
            if 'bufferView' in g['images'][i]:
                used_view.add(g['images'][i]['bufferView'])
        view_order, view_map = remap(len(g.get('bufferViews', [])), used_view)

        # rewrite references
        for m in g.get('meshes', []):
            for p in m['primitives']:
                p['attributes'] = {k: acc_map[v] for k, v in p['attributes'].items()}
                if 'indices' in p:
                    p['indices'] = acc_map[p['indices']]
                if 'targets' in p:
                    p['targets'] = [{k: acc_map[v] for k, v in t.items()} for t in p['targets']]
        for s in g.get('skins', []):
            if 'inverseBindMatrices' in s:
                s['inverseBindMatrices'] = acc_map[s['inverseBindMatrices']]
        for a in g.get('animations', []):
            for smp in a['samplers']:
                smp['input'] = acc_map[smp['input']]
                smp['output'] = acc_map[smp['output']]

        def fix_tex(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k.endswith('Texture') and isinstance(v, dict) and 'index' in v:
                        v['index'] = tex_map[v['index']]
                    fix_tex(v)
            elif isinstance(o, list):
                for v in o:
                    fix_tex(v)
        fix_tex(g.get('materials', []))
        textures = [g['textures'][i] for i in tex_order]
        for t in textures:
            if 'source' in t:
                t['source'] = img_map[t['source']]
            for ext in t.get('extensions', {}).values():
                if 'source' in ext:
                    ext['source'] = img_map[ext['source']]
        accessors = [g['accessors'][i] for i in acc_order]
        for a in accessors:
            if 'bufferView' in a:
                a['bufferView'] = view_map[a['bufferView']]
            if 'sparse' in a:
                a['sparse']['indices']['bufferView'] = view_map[a['sparse']['indices']['bufferView']]
                a['sparse']['values']['bufferView'] = view_map[a['sparse']['values']['bufferView']]
        images = [g['images'][i] for i in img_order]
        for im in images:
            if 'bufferView' in im:
                im['bufferView'] = view_map[im['bufferView']]
        old_views = g.get('bufferViews', [])
        g['accessors'], g['images'], g['textures'] = accessors, images, textures
        for k in ('accessors', 'images', 'textures'):
            if not g[k]:
                del g[k]
        return [old_views[i] for i in view_order]

    def write(self) -> bytes:
        views = self._prune()
        out = bytearray()
        for v in views:
            off = v['byteOffset'] if 'byteOffset' in v else 0
            if off < 0:
                payload = self.extra[-1 - off]
            else:
                payload = bytes(self.binary[off: off + v['byteLength']])
            out += b'\0' * ((-len(out)) % 4)  # 4-byte alignment for every view
            v['byteOffset'] = len(out)
            out += payload
        out += b'\0' * ((-len(out)) % 4)
        self.gltf['bufferViews'] = views
        self.gltf['buffers'] = [{'byteLength': len(out)}]
        js = json.dumps(self.gltf, separators=(',', ':')).encode('utf-8')
        js += b' ' * ((-len(js)) % 4)
        total = 12 + 8 + len(js) + 8 + len(out)
        return (struct.pack('<III', GLB_MAGIC, 2, total) + struct.pack('<II', len(js), CHUNK_JSON) + js
                + struct.pack('<II', len(out), CHUNK_BIN) + bytes(out))


def replace_primitive_geometry(editor: GlbEditor, binary: bytearray, surf: Surface,
                               vmap: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> None:
    """
    After a fresh unwrap: rebuild one primitive from `vmap` (new vertex -> old vertex), the
    unwrap's faces and UVs. Every vertex attribute and morph target is gathered through
    vmap, so skin weights and normals follow the split vertices; the bake's UV set replaces
    TEXCOORD_<n>.
    """
    g = editor.gltf
    prim = g['meshes'][surf.mesh]['primitives'][surf.prim]
    new_attrs = {}
    for name, ai in prim['attributes'].items():
        acc = g['accessors'][ai]
        if name == f'TEXCOORD_{surf.texcoord}':
            continue
        data = read_accessor(g, binary, ai) if not acc.get('normalized') else None
        if data is None:
            # keep normalized integer storage as it was (colours, weights)
            dtype = np.dtype(COMPONENT[acc['componentType']])
            raw = read_accessor({**g, 'accessors': [{**acc, 'normalized': False}]}, binary, 0)
            new_attrs[name] = editor.add_accessor(raw[vmap].astype(dtype), acc['type'], acc['componentType'], 34962, normalized=True)
        else:
            new_attrs[name] = editor.add_accessor(data[vmap], acc['type'], acc['componentType'], 34962, minmax=(name == 'POSITION'))
    new_attrs[f'TEXCOORD_{surf.texcoord}'] = editor.add_accessor(uvs.astype(np.float32), 'VEC2', 5126, 34962)
    prim['attributes'] = new_attrs
    if 'targets' in prim:
        prim['targets'] = [
            {k: editor.add_accessor(read_accessor(g, binary, v)[vmap], g['accessors'][v]['type'], 5126, 34962, minmax=True)
             for k, v in t.items()} for t in prim['targets']]
    n = int(vmap.shape[0])
    comp = 5125 if n > 65535 else 5123
    prim['indices'] = editor.add_accessor(faces.reshape(-1, 1).astype(np.int64), 'SCALAR', comp, 34963)
