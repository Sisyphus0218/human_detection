from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray  # Follows OpenCV's BGR channel order.
    depth_mm: np.ndarray | None
    frame_index: int
    timestamp_ms: float
