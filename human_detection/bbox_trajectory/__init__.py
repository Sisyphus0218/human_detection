"""Target bounding-box trajectory tracking and prediction."""

from .bbox_trajectory_entry import BBoxSource, BBoxTrajectoryEntry
from .target_bbox_trajectory import TargetBBoxTrajectory

__all__ = ["BBoxSource", "BBoxTrajectoryEntry", "TargetBBoxTrajectory"]
