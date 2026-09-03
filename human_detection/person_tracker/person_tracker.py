from abc import ABC, abstractmethod

import numpy as np

from .tracked_person import TrackedPerson


class PersonTracker(ABC):
    """Common interface for person trackers."""

    @abstractmethod
    def track(
        self,
        frame: np.ndarray,
    ) -> list[TrackedPerson]:
        """Track people in one frame; return an empty list when none are found."""
