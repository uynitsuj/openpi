import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import List
from PIL import Image

class FrozenCLIPEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(ckpt).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(ckpt)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str], length B
        returns: (B, 512) CLIP text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        return text_embeds

    def encode_image(self, images: List[Image.Image], do_rescale=False) -> torch.Tensor:
        """
        images: list of PIL Images, length B
        returns: (B, 512) CLIP image embeddings
        """
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=do_rescale).to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return image_embeds
