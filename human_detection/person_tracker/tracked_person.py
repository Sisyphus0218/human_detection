from dataclasses import dataclass

from human_detection.utils import BBox


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: BBox
    confidence: float
