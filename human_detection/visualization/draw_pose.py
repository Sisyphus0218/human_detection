import cv2
import numpy as np

from human_detection.pose_estimator import PoseEstimationResult

POSE_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def draw_pose(frame: np.ndarray, pose: PoseEstimationResult | None) -> None:
    """Draw available current-frame keypoints and their skeleton connections."""
    if pose is None:
        return
    points = {
        name: (int(round(keypoint.position_2d.u)), int(round(keypoint.position_2d.v)))
        for name, keypoint in pose.keypoints.items()
        if np.isfinite(keypoint.position_2d.as_tuple()).all()
    }
    for first, second in POSE_CONNECTIONS:
        if first in points and second in points:
            cv2.line(frame, points[first], points[second], (0, 255, 255), 2)
    for point in points.values():
        cv2.circle(frame, point, 4, (0, 0, 255), -1)
