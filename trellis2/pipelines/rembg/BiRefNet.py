# File: trellis2/pipelines/rembg/BiRefNet.py
# trellis2/pipelines/rembg/BiRefNet.py
from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import os # <--- Added

class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        
        # 1. Check relative to Current Working Directory
        cwd_path = os.path.join(os.getcwd(), "MODELS", "RMBG-2.0")
        
        # 2. Check relative to this script file
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "MODELS", "RMBG-2.0"))
        
        target_path = None
        use_local = False

        # We look for model.safetensors or config.json to confirm validity
        if os.path.exists(os.path.join(cwd_path, "config.json")):
            target_path = cwd_path
            use_local = True
        elif os.path.exists(os.path.join(script_path, "config.json")):
            target_path = script_path
            use_local = True
        else:
            # Fallback to whatever was passed in (likely briaai/RMBG-2.0 which will fail if not logged in)
            target_path = model_name
            use_local = False
            
        if use_local:
            print(f"[INFO] Local RMBG-2.0 found. Loading from: {target_path}")
        else:
            print(f"[INFO] Local RMBG-2.0 NOT found. Attempting download/access: {model_name}")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            target_path, 
            trust_remote_code=True,
            local_files_only=use_local # Forces offline mode if local files are found
        )
        
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image