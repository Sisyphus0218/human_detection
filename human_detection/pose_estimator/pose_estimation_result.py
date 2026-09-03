from dataclasses import dataclass


@dataclass(frozen=True)
class PoseKeypoint:
    """One pose keypoint expressed in full-frame pixel coordinates."""

    name: str
    position_2d: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class PoseEstimationResult:
    """2D pose-estimation result for one person."""

    keypoints: dict[str, PoseKeypoint]

    def get(self, name: str) -> PoseKeypoint | None:
        return self.keypoints.get(name)
