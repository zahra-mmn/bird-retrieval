"""Thin wrapper around Meta's ImageBind for encoding text/image/audio into the shared 1024-dim
space. Install ImageBind from GitHub, not PyPI — see README (there is no `imagebind` package on
PyPI; the original scope doc's `pip install imagebind` would fail).
"""

import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


class ImageBindEncoder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = {ModalityType.TEXT: data.load_and_transform_text(texts, self.device)}
        return self.model(inputs)[ModalityType.TEXT]

    @torch.no_grad()
    def encode_images(self, image_paths: list[str]) -> torch.Tensor:
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        return self.model(inputs)[ModalityType.VISION]

    @torch.no_grad()
    def encode_audio(self, audio_paths: list[str]) -> torch.Tensor:
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        return self.model(inputs)[ModalityType.AUDIO]

    def encode_modality(self, modality: str, items: list[str]) -> torch.Tensor:
        return {
            "text": self.encode_text,
            "image": self.encode_images,
            "audio": self.encode_audio,
        }[modality](items)
