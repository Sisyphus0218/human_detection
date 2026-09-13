from dataclasses import dataclass

from human_detection.follow_controller.robot_velocity import VelocityCommand
from human_detection.frame_source import RGBDFrame
from human_detection.pose_estimator import PoseEstimationResult
from human_detection.utils import BBox, Point3D


@dataclass(frozen=True)
class TrackingFrameResult:
    """Current-frame observations; target_bbox never contains a predicted box."""

    rgbd_frame: RGBDFrame
    target_bbox: BBox | None
    pose: PoseEstimationResult | None
    position: Point3D | None
    velocity_command: VelocityCommand | None = None
