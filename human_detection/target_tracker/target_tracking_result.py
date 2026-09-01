from dataclasses import dataclass

from human_detection.person_tracker import PersonTrackingResult, TrackedPerson

from .target_bbox_trajectory import BBoxTrajectoryEntry


@dataclass
class TargetTrackingResult:
    frame_index: int
    person_tracking_result: PersonTrackingResult
    target: TrackedPerson | None
    trajectory_entry: BBoxTrajectoryEntry
    state: str
