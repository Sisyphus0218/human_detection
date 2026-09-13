from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point3D:
    """A 3D point in millimeters.

    The coordinate frame must be specified by the calling interface, such as
    camera or robot body; this class does not store or transform that frame.
    """

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def midpoint(self, other: Point3D) -> Point3D:
        """Return the midpoint; both points must use the same coordinate frame."""
        return Point3D(
            (self.x + other.x) / 2,
            (self.y + other.y) / 2,
            (self.z + other.z) / 2,
        )
