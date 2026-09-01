from abc import ABC, abstractmethod

import numpy as np

from .person_tracking_result import PersonTrackingResult


class PersonTracker(ABC):
    """Common interface for person trackers."""

    @abstractmethod
    def track_frame(
        self,
        frame: np.ndarray,
    ) -> PersonTrackingResult:
        """Track people in one frame and return the tracking result."""
