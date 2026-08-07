from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor


class PersonFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        model_path: str | Path,
        device: str = "cpu",
    ):
        self.feature_extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(model_path),
            device=device,
            verbose=False,
        )

    @torch.inference_mode()
    def extract_features(
        self,
        bgr_images: np.ndarray | list[np.ndarray],
    ) -> torch.Tensor:
        if isinstance(bgr_images, np.ndarray):
            bgr_images = [bgr_images]

        rgb_images = []
        for index, image in enumerate(bgr_images):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_image)

        features = self.feature_extractor(rgb_images)
        return F.normalize(features, p=2, dim=1)
