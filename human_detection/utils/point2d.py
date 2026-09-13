from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    """An image point in pixels, with u rightward and v downward."""

    u: float
    v: float

    def as_tuple(self) -> tuple[float, float]:
        return self.u, self.v

    def midpoint(self, other: Point2D) -> Point2D:
        """Return the midpoint of two points in the same image coordinates."""
        return Point2D((self.u + other.u) / 2, (self.v + other.v) / 2)
