from dataclasses import dataclass

from human_detection.person_tracker import TrackedPerson


@dataclass(frozen=True)
class TargetTrackingResult:
    target: TrackedPerson | None
    state: str
