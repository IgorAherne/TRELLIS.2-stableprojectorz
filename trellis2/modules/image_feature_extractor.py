# trellis2/modules/image_feature_extractor.py
import os
from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image

class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        
        # Check if we have bundled the model in the MODELS/dinov3 folder
        # We use absolute path relative to the current working directory
        local_dir = os.path.join(os.getcwd(), "MODELS", "dinov3")
        if os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"[INFO] Gated Repo Redirect: Loading DINOv3 from {local_dir}")
            model_to_load = local_dir
        else:
            model_to_load = model_name

        self.model = DINOv3ViTModel.from_pretrained(model_to_load)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        
        # 1. Check relative to Current Working Directory (Root of repo)
        cwd_path = os.path.join(os.getcwd(), "MODELS", "dinov3")
        
        # 2. Check relative to this script file (Backup in case cwd changes)
        # file is in trellis2/modules/, so we go up two levels to reach root
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MODELS", "dinov3"))
        
        target_path = None
        use_local = False

        if os.path.exists(os.path.join(cwd_path, "config.json")):
            target_path = cwd_path
            use_local = True
        elif os.path.exists(os.path.join(script_path, "config.json")):
            target_path = script_path
            use_local = True
        else:
            target_path = model_name
            use_local = False
            
        if use_local:
            print(f"[INFO] Local DINOv3 found. Loading from: {target_path}")
        else:
            print(f"[INFO] Local DINOv3 NOT found. Attempting download: {model_name}")

        # We pass local_files_only=use_local to strict enforce not touching the internet
        self.model = DINOv3ViTModel.from_pretrained(target_path, local_files_only=use_local)
        
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.extract_features(image)
        return features
