from abc import ABC, abstractmethod

import numpy as np

from human_detection.utils import BBox

from .pose_estimation_result import PoseEstimationResult


class PoseEstimator(ABC):
    """Common interface for estimating one tracked person's 2D pose."""

    @abstractmethod
    def estimate(
        self,
        frame_bgr: np.ndarray,
        bbox: BBox | None,
    ) -> PoseEstimationResult | None:
        """
        Estimate pose keypoints inside bbox, or return None when bbox is None.

        Returned keypoints use pixel coordinates in the complete input frame,
        not normalized coordinates or coordinates relative to the crop.
        """
