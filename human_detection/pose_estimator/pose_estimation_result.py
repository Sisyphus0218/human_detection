from dataclasses import dataclass

from human_detection.utils import Point2D


@dataclass(frozen=True)
class PoseKeypoint:
    """One pose keypoint expressed in full-frame pixel coordinates."""

    name: str
    position_2d: Point2D
    confidence: float


@dataclass(frozen=True)
class PoseEstimationResult:
    """2D pose-estimation result for one person."""

    keypoints: dict[str, PoseKeypoint]

    def get(self, name: str) -> PoseKeypoint | None:
        return self.keypoints.get(name)
