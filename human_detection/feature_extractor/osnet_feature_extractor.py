from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor as TorchReIDFeatureExtractor

from .feature_extractor import FeatureExtractor


class OSNetFeatureExtractor(FeatureExtractor):
    """Extract normalized person features with an OSNet model."""

    def __init__(
        self,
        model_name: str,
        model_path: str | Path,
        feature_dim: int = 512,
        device: str = "cpu",
    ) -> None:
        self._feature_dim = feature_dim
        self._extractor = TorchReIDFeatureExtractor(
            model_name=model_name,
            model_path=str(model_path),
            device=device,
            verbose=False,
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @torch.inference_mode()
    def extract_features(
        self,
        bgr_images: np.ndarray | list[np.ndarray],
    ) -> torch.Tensor:
        """
        Extract person features from one BGR image or a list of BGR images.

        Args:
            bgr_images: A BGR image or a list of BGR images.

        Returns:
            Normalized features with shape ``(N, feature_dim)``.
        """

        if isinstance(bgr_images, np.ndarray):
            bgr_images = [bgr_images]
        if not bgr_images:
            raise ValueError("No images provided for feature extraction.")

        rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in bgr_images]
        features = self._extractor(rgb_images)
        return F.normalize(features, p=2, dim=1)
