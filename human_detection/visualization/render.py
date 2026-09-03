from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from human_detection.bbox_trajectory import BBoxSource
from .draw_pose import draw_pose

if TYPE_CHECKING:
    from human_detection.pipeline.tracking_frame_result import TrackingFrameResult


def _draw_target_tracking(
    frame: np.ndarray,
    result: TrackingFrameResult,
) -> None:
    """Draw the selected target or its predicted bounding box in place."""
    target = result.target_result.target
    if target is not None:
        x1, y1, x2, y2 = target.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"Target ID: {target.track_id}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    elif (
        result.trajectory_entry.source is BBoxSource.PREDICTED
        and result.trajectory_entry.bbox is not None
    ):
        x1, y1, x2, y2 = result.trajectory_entry.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.putText(
            frame,
            "Target predicted",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

    cv2.putText(
        frame,
        result.target_result.state,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def render_tracking_frame(result: TrackingFrameResult) -> np.ndarray:
    """Render the target and all enabled algorithm results."""
    frame = result.rgbd_frame.color_bgr.copy()
    _draw_target_tracking(frame, result)
    draw_pose(frame, result.pose)

    if result.position_mm is not None:
        x_mm, y_mm, z_mm = result.position_mm
        cv2.putText(
            frame,
            f"Position: ({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f}) mm",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    return frame


def render_debug_frame(result: TrackingFrameResult) -> np.ndarray:
    """Render only bounding boxes for all tracked people."""
    frame = result.rgbd_frame.color_bgr.copy()

    for person in result.tracked_persons:
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return frame
