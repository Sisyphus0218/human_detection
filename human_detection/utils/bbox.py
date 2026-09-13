from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """An xyxy box in source-image pixels; right/bottom bounds are exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid bbox: ({self.x1}, {self.y1}, "
                f"{self.x2}, {self.y2}); "
                "expected x2 >= x1 and y2 >= y1"
            )

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def calculate_iou(self, other: BBox) -> float:
        """Calculate intersection over union with another box."""
        intersection_width = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        intersection_height = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection_area = intersection_width * intersection_height

        union_area = self.area + other.area - intersection_area
        if union_area <= 0:
            return 0.0

        return intersection_area / union_area
