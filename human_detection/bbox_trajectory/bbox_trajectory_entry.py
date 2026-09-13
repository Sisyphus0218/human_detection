from dataclasses import dataclass
from enum import Enum

from human_detection.utils import BBox


class BBoxSource(Enum):
    MISSING = "missing"
    PREDICTED = "predicted"
    OBSERVED = "observed"


@dataclass(frozen=True)
class BBoxTrajectoryEntry:
    frame_index: int
    bbox: BBox | None
    source: BBoxSource
