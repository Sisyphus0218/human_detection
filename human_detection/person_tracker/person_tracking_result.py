from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class PersonTrackingResult:
    frame: np.ndarray
    tracked_persons: list[TrackedPerson]
