from dataclasses import dataclass


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
