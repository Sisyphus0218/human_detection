from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class PersonDetector(ABC):
    """Common interface for person detectors."""

    @abstractmethod
    def detect_target(
        self,
        target_image_dir: str | Path,
        target_crop_dir: str | Path,
    ) -> list[np.ndarray]:
        """
        Crop the target from each image in the target_image_dir.
        Save the cropped images to target_crop_dir.

        Args:
            target_image_dir: Directory containing the images to process.
            target_crop_dir: Directory to save the cropped images.

        Returns:
            crops: A list of cropped BGR images.
        """
