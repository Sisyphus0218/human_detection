from dataclasses import dataclass

from human_detection.bbox_trajectory import BBoxTrajectoryEntry
from human_detection.frame_source import RGBDFrame
from human_detection.person_tracker import TrackedPerson
from human_detection.pose_estimator import PoseEstimationResult
from human_detection.position_estimator import PositionEstimationResult
from human_detection.target_tracker import TargetTrackingResult


@dataclass(frozen=True)
class TrackingFrameResult:
    rgbd_frame: RGBDFrame
    tracked_persons: list[TrackedPerson]
    target_result: TargetTrackingResult
    trajectory_entry: BBoxTrajectoryEntry
    pose: PoseEstimationResult | None
    position: PositionEstimationResult | None
