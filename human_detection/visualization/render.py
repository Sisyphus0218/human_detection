import cv2
import numpy as np

from human_detection.target_tracker import BBoxSource, TargetTrackingResult


def render_tracking_frame(result: TargetTrackingResult) -> np.ndarray:
    """Render the selected target or its predicted bounding box."""
    frame = result.person_tracking_result.frame.copy()

    if result.target is not None:
        x1, y1, x2, y2 = result.target.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"Target ID: {result.target.track_id}",
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
        result.state,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return frame


def render_debug_frame(result: TargetTrackingResult) -> np.ndarray:
    """Render all tracked people and the selected target state."""
    frame = result.person_tracking_result.frame.copy()
    selected_id = result.target.track_id if result.target is not None else None

    for person in result.person_tracking_result.tracked_persons:
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {person.track_id} | Conf: {person.confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    frame_height, frame_width = frame.shape[:2]
    panel_top = max(0, frame_height - 44)
    panel = frame.copy()
    cv2.rectangle(panel, (0, panel_top), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)
    cv2.putText(
        frame,
        f"Frame {result.frame_index} | State: {result.state} | Selected ID: {selected_id}",
        (12, panel_top + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    if result.target is not None:
        x1, y1, x2, y2 = result.target.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
        cv2.putText(
            frame,
            f"ReID selected ID: {result.target.track_id}",
            (x1, max(y1 - 12, 145)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )

    return frame
