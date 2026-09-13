from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityFeedback:
    """Measured planar robot velocity expressed in the robot body frame."""

    vx: float
    vy: float
    wz: float
    timestamp_ms: float
    valid: bool = True


@dataclass(frozen=True)
class VelocityCommand:
    """Desired robot velocity expressed in the robot body frame."""

    vx: float
    vy: float
    wz: float
    valid: bool
