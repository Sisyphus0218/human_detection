from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Point3D:
    """A 3D point in millimeters.

    The coordinate frame must be specified by the calling interface, such as
    camera or robot body; this class does not store the frame.
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

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> Point3D:
        """Return a new point using p_target = rotation @ p_source + translation.

        rotation must map the source frame to the target frame and have shape
        (3, 3). translation must have shape (3,) and be in millimeters.
        """
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)

        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")

        position = rotation @ np.asarray(self.as_tuple()) + translation

        return Point3D(float(position[0]), float(position[1]), float(position[2]))
