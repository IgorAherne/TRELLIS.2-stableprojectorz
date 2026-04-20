import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipeline.cuda()

image = Image.open('assets/example_image/0f168a4b1b6e96c72e9627c97a212c27a4572250ff58e25703b9d0c2bc74191a.webp')
image = pipeline.preprocess_image(image)

t0 = time.time()
outputs, latents = pipeline.run(
    image, seed=42, preprocess_image=False,
    pipeline_type='512', return_latent=True,
)
print(f"\nTotal: {time.time()-t0:.1f}s")