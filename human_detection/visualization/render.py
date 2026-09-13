from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from human_detection.utils import BBox, Point3D
from .draw_pose import draw_pose

if TYPE_CHECKING:
    from human_detection.pipeline.tracking_frame_result import TrackingFrameResult


def draw_bbox(frame: np.ndarray, bbox: BBox | None) -> None:
    """Draw a current observed box using exclusive right/bottom bounds."""
    if bbox is None:
        return
    height, width = frame.shape[:2]
    x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
    x2, y2 = min(width, bbox.x2), min(height, bbox.y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)


def draw_position(frame: np.ndarray, position: Point3D | None) -> None:
    """Display the current camera-space position, without reusing old values."""
    text = "Position (camera, mm): unavailable"
    if position is not None:
        text = (
            f"Position (camera, mm): X={position.x:.0f} "
            f"Y={position.y:.0f} Z={position.z:.0f}"
        )
    cv2.putText(
        frame,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render_tracking_frame(
    result: TrackingFrameResult,
    *,
    render_bbox: bool = True,
    render_pose: bool = True,
    render_position: bool = True,
) -> np.ndarray:
    """Render enabled overlays on a copy for both display and video output."""
    frame = result.rgbd_frame.color_bgr.copy()
    if render_bbox:
        draw_bbox(frame, result.target_bbox)
    if render_pose:
        draw_pose(frame, result.pose)
    if render_position:
        draw_position(frame, result.position)
    return frame
