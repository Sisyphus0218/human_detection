from abc import ABC, abstractmethod

import numpy as np
import torch


class FeatureExtractor(ABC):
    """Common interface for person feature extractors."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimension of one feature vector."""

    @abstractmethod
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
