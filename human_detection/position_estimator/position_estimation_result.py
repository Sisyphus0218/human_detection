from dataclasses import dataclass


@dataclass(frozen=True)
class PositionEstimationResult:
    """Body positions in the camera coordinate system, in millimeters."""

    neck: tuple[float, float, float] | None = None
    hip: tuple[float, float, float] | None = None
    knee: tuple[float, float, float] | None = None
    ankle: tuple[float, float, float] | None = None
