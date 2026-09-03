from dataclasses import dataclass
from enum import Enum


class BBoxSource(Enum):
    MISSING = "missing"
    PREDICTED = "predicted"
    OBSERVED = "observed"


@dataclass(frozen=True)
class BBoxTrajectoryEntry:
    frame_index: int
    bbox: tuple[int, int, int, int] | None
    source: BBoxSource
    track_id: int | None
