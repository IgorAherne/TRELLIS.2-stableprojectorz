# File: make_fp16_twins.py
# make_fp16_twins.py
"""
Optional: write fp16 twins of TRELLIS.2's bf16 checkpoints, so the loader can map them
read-only. Nothing requires it; without twins the flow models are simply held in RAM.

    python make_fp16_twins.py              # write the twins (run from the repo root, venv active)
    python make_fp16_twins.py --check      # list what would be written
    python make_fp16_twins.py --remove     # delete the twins

Why: the loader (trellis2/models/__init__.py) keeps weights as read-only views of their
files, which Windows does not charge to commit. But the five flow models are bf16 on disk and
the pipeline converts them to fp16 in RAM, so a view of the bf16 file would still be copied.
This builds each such model exactly as the pipeline does (lean load, then bf16 -> fp16, the
same .half() as _convert_bf16_to_fp16_if_needed) and saves the result next to the checkpoint
as <name>.fp16.safetensors. The loader then maps that file instead: the same bits the pipeline
held in RAM before, now paged from disk. Each twin is ~2.4 GB (~12 GB for all five).

A twin records the size of the checkpoint it came from; if the checkpoint changes, the
loader ignores the twin until this is re-run.
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


def checkpoints(models_dir):
    """(pipeline key, model path without extension, .safetensors file) for every model."""
    snaps = glob.glob(os.path.join(models_dir, 'hub', 'models--microsoft--TRELLIS.2-4B', 'snapshots', '*', 'pipeline.json'))
    if not snaps:
        sys.exit(f"no TRELLIS.2-4B pipeline.json under {models_dir}{os.sep}hub (pass --models <HF_HOME>)")
    out = []
    for pj in snaps:
        snap = os.path.dirname(pj)
        with open(pj) as f:
            models = json.load(f)['args']['models']
        for key, rel in models.items():
            f = os.path.join(snap, rel + '.safetensors')
            if os.path.exists(f):
                out.append((key, os.path.join(snap, rel), f))
    return out


def twin_of(f):
    return f[:-len('.safetensors')] + '.fp16.safetensors'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=os.environ.get('HF_HOME') or os.path.join(ROOT, 'models'),
                    help='the Hugging Face cache holding the weights (default: $HF_HOME, else ./models)')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--remove', action='store_true')
    a = ap.parse_args()

    todo = []
    for key, mpath, f in checkpoints(a.models):
        dtypes = {v['dtype'] for k, v in header(f).items() if k != '__metadata__'}
        if 'BF16' in dtypes:
            todo.append((key, mpath, f))

    if a.remove:
        for _, _, f in todo:
            t = twin_of(f)
            if os.path.exists(t):
                os.remove(t)
                print(f"removed  {os.path.basename(t)}")
        return

    # Build the models with the lean loader but no mapping, so they are read from the
    # original checkpoint into RAM exactly as before.
    os.environ['TRELLIS_MMAP_WEIGHTS'] = '0'
    sys.path.insert(0, ROOT)
    import torch
    from safetensors.torch import save_file
    import trellis2.models as models

    for key, mpath, f in todo:
        t = twin_of(f)
        size = os.path.getsize(f)
        if os.path.exists(t):
            try:
                if int(header(t).get('__metadata__', {}).get('source_bytes', -1)) == size:
                    print(f"up to date  {key:<28} {os.path.basename(t)}")
                    continue
            except Exception:
                pass
        if a.check:
            print(f"would write {key:<28} {os.path.basename(t)}")
            continue
        t0 = time.time()
        model = models.from_pretrained(mpath)
        for m in model.modules():
            for name, p in m._parameters.items():
                if p is not None and p.dtype == torch.bfloat16:
                    p.data = p.data.half()
            for name, b in m._buffers.items():
                if b is not None and b.dtype == torch.bfloat16:
                    m._buffers[name] = b.half()
        sd = model.state_dict()
        keys = [k for k in header(f) if k != '__metadata__']
        missing = [k for k in keys if k not in sd]
        if missing:
            print(f"  note: {len(missing)} checkpoint keys are not in the model (e.g. {missing[0]}); left out")
        tensors = {k: sd[k].contiguous() for k in keys if k in sd}
        left = [k for k, v in tensors.items() if v.dtype == torch.bfloat16]
        if left:
            sys.exit(f"{key}: {len(left)} tensors still bf16 after conversion ({left[0]}) - not writing")
        tmp = t + '.tmp'
        save_file(tensors, tmp, metadata={'source': os.path.basename(f), 'source_bytes': str(size),
                                          'note': 'weights as the pipeline holds them after bf16->fp16; see make_fp16_twins.py'})
        os.replace(tmp, t)
        del model, sd, tensors
        print(f"wrote       {key:<28} {os.path.basename(t)}  ({os.path.getsize(t) / 2**30:.2f} GB, {time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
