# File: trellis2/models/__init__.py
# trellis2/models/__init__.py
import importlib
import threading

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    # Lean load. The original path builds every weight in fp32, converts it to half,
    # then loads a second full copy from the checkpoint - about 2.4x the model's real
    # size stays committed afterwards. Build the parameters on the meta device instead and
    # assign the checkpoint's tensors (cast to the dtype the model expects). Same weights,
    # same dtypes. Falls back to the original path on any surprise. TRELLIS_LEAN_LOAD=0 opts out.
    if os.environ.get('TRELLIS_LEAN_LOAD', '1') != '0':
        model = _lean_load(__getattr__(config['name']), config['args'], kwargs, model_file)
        if model is not None:
            return model
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


def mmap_enabled(path):
    """
    Keep weights as read-only views of their files instead of copies in RAM
    (TRELLIS_MMAP_WEIGHTS, on by default). Not for files on a network share, which would then
    page in over the network mid-generation; those are copied as before.
    """
    import os
    if os.environ.get('TRELLIS_MMAP_WEIGHTS', '1') == '0':
        return False
    p = os.path.abspath(path)
    if p.startswith(os.sep * 2):
        return False
    try:
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(os.path.splitdrive(p)[0] + os.sep) == 4:  # DRIVE_REMOTE
            return False
    except Exception:
        pass
    return True


def fp16_twin(path):
    """
    <name>.fp16.safetensors beside a checkpoint (written by make_fp16_twins.py) holds the
    weights exactly as the pipeline keeps them after its bf16->fp16 pass. It is used only
    while its recorded source size still matches the checkpoint.
    """
    import os, json, struct
    stem = path[:-len('.safetensors')] if path.endswith('.safetensors') else path
    twin = stem + '.fp16.safetensors'
    if not os.path.exists(twin):
        return None
    try:
        with open(twin, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            meta = json.loads(f.read(n)).get('__metadata__', {})
        if int(meta.get('source_bytes', -1)) != os.path.getsize(path):
            print(f"[MMAP] {os.path.basename(twin)} is stale (checkpoint changed); re-run make_fp16_twins.py")
            return None
    except Exception as e:
        print(f"[MMAP] cannot read {os.path.basename(twin)} ({e})")
        return None
    return twin


def memmap_safetensors(path):
    """
    The tensors of a .safetensors file as read-only views of a numpy memmap.
    On Windows, safetensors' own mapping is charged to commit at full size (measured: a
    2.4 GB file adds 2.4 GB of private bytes before anything is read), and a copy is charged
    as well. A read-only mapping is charged nothing: its pages are file-backed, so under
    pressure the OS drops them instead of writing them to the pagefile, and reads them back
    from disk the next time a stage moves that model to the GPU.
    The views must never be written to (an in-place write is an access violation).
    """
    import json, struct, warnings
    import numpy as np
    import torch
    dtypes = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
              'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8,
              'U8': torch.uint8, 'BOOL': torch.bool}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode='r')
    base = 8 + n
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # "The given NumPy array is not writable"
        for k, info in header.items():
            if k == '__metadata__':
                continue
            a, b = info['data_offsets']
            dt = dtypes[info['dtype']]
            if b == a:
                out[k] = torch.empty(info['shape'], dtype=dt)
                continue
            raw = torch.from_numpy(mm[base + a: base + b])
            try:
                out[k] = raw.view(dt).view(info['shape'])
            except RuntimeError:
                # Misaligned for its dtype: this one tensor is copied instead.
                out[k] = raw.clone().view(dt).view(info['shape'])
    return out


def memmap_rebind(module, path):
    """
    Point a loaded module's parameters at read-only views of its checkpoint (same name,
    shape and dtype only), so the copy its own loader made can be freed. Used for DINOv3,
    which transformers loads.
    """
    import os
    if not os.path.exists(path) or not mmap_enabled(path):
        return
    try:
        views = memmap_safetensors(path)
    except Exception as e:
        print(f"[MMAP] {os.path.basename(path)} not mapped ({e})")
        return
    hit = miss = 0
    for name, p in module.named_parameters():
        v = views.get(name)
        if v is None:
            # transformers may prefix the checkpoint's keys with the model's base name
            v = next((views[k] for k in views if k.endswith('.' + name)), None)
        if v is not None and v.shape == p.shape and v.dtype == p.dtype and p.device.type == 'cpu':
            p.data = v
            hit += 1
        else:
            miss += 1
    print(f"[MMAP] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: {hit} parameters mapped, {miss} kept")


_LEAN_BUILD_LOCK = threading.Lock()


def _lean_load(cls, args, kwargs, model_file):
    import os
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    # The build below patches nn.Module.register_parameter and cls.device for the whole
    # process. One lean build at a time, and both patches act only on this thread, so a
    # module constructed elsewhere meanwhile is built exactly as before.
    builder = threading.get_ident()
    _LEAN_BUILD_LOCK.acquire()
    # Parameters go to meta as they are registered; buffers and plain tensor attributes
    # (RoPE frequency tables, etc.) are still built for real, exactly as before.
    register = nn.Module.register_parameter
    def register_on_meta(self, name, param):
        if threading.get_ident() == builder and param is not None and not param.is_meta:
            param = nn.Parameter(param.to('meta'), requires_grad=param.requires_grad)
        register(self, name, param)
    nn.Module.register_parameter = register_on_meta
    # Large factory calls (torch.empty/zeros/randn/... - how layers allocate their weights)
    # go straight to meta too. Otherwise each weight is allocated on the CPU first and
    # freed, and PyTorch's Windows allocator (mimalloc) keeps that memory committed: ~1.5 GB
    # per decoder that nothing uses. Computed tensors (rope_phases etc.) are not factory
    # calls and stay real; anything that ends up meta anyway is caught below.
    from torch.overrides import TorchFunctionMode
    factories = {torch.empty, torch.zeros, torch.ones, torch.full, torch.randn, torch.rand}
    class BigFactoriesOnMeta(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in factories and kwargs.get('device') is None and args:
                size = args[0] if func is torch.full or isinstance(args[0], (tuple, list, torch.Size)) else args
                try:
                    numel = 1
                    for s in size:
                        numel *= int(s)
                except TypeError:
                    numel = 0
                if numel >= 65536:
                    kwargs = dict(kwargs, device='meta')
            return func(*args, **kwargs)
    # Some constructors build buffers on `self.device` (the first parameter's device, e.g.
    # SparseStructureFlowModel's rope_phases). The original build always runs on the CPU, so
    # pin that property to the CPU while constructing.
    device_prop = cls.__dict__.get('device') if isinstance(cls.__dict__.get('device'), property) else None
    if device_prop is not None:
        cls.device = property(lambda self: torch.device('cpu') if threading.get_ident() == builder
                              else device_prop.fget(self))
    try:
        with BigFactoriesOnMeta():  # torch function modes are per thread already
            model = cls(**args, **kwargs)
    except Exception as e:
        print(f"[LEAN LOAD] building {cls.__name__} on meta failed ({e}); using the original loader")
        return None
    finally:
        nn.Module.register_parameter = register
        if device_prop is not None:
            cls.device = device_prop
        _LEAN_BUILD_LOCK.release()

    target = dict(model.named_parameters())
    target.update(model.named_buffers())
    if mmap_enabled(model_file):
        # Read-only file views: no commit charge, and the OS can drop the pages under memory
        # pressure and read them back from disk (see memmap_safetensors). A tensor is used as
        # a view when its dtype is final; bf16 checkpoints only get there through their fp16
        # twin (make_fp16_twins.py), since the pipeline turns bf16 into fp16.
        twin = fp16_twin(model_file)
        src = twin or model_file
        mapped = memmap_safetensors(src)
        sd = {}
        views = copies = 0
        bf16 = False
        for k in list(mapped):
            v = mapped.pop(k)
            want = target[k].dtype if k in target else v.dtype
            bf16 = bf16 or v.dtype == torch.bfloat16
            if v.dtype == want or (twin and want == torch.bfloat16 and v.dtype == torch.float16):
                sd[k] = v
                views += 1
            else:
                # e.g. an fp16 checkpoint tensor for a layer the model keeps in fp32
                sd[k] = v.to(want)
                copies += 1
        del mapped
        hint = ' (bf16 checkpoint: run make_fp16_twins.py to map it too)' if bf16 and not twin else ''
        print(f"[MMAP] {os.path.basename(src)}: {views} mapped, {copies} copied{hint}")
    else:
        # load_file's tensors are views of a memory-mapped checkpoint.
        # Copy each into ordinary memory, one at a time, so the model never pages from the
        # file (which matters when the weights live on a network share or a slow disk).
        mapped = load_file(model_file)
        sd = {}
        for k in list(mapped):
            v = mapped.pop(k)
            want = target[k].dtype if k in target else v.dtype
            sd[k] = v.to(want) if v.dtype != want else v.clone()
            del v
        del mapped
    try:
        model.load_state_dict(sd, strict=False, assign=True)
    except Exception as e:
        print(f"[LEAN LOAD] {os.path.basename(model_file)} did not load ({e}); using the original loader")
        return None
    del sd
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        # The checkpoint does not cover every weight; the original path keeps random init there.
        print(f"[LEAN LOAD] {os.path.basename(model_file)} leaves {len(left)} weights unset ({left[0]}, ...); using the original loader")
        return None
    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
