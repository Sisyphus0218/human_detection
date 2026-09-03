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


def draw_pose(
    frame: np.ndarray,
    pose_result: PoseEstimationResult | None,
) -> None:
    """Draw one pose on frame in place."""
    if pose_result is None:
        return

    for first_name, second_name in POSE_CONNECTIONS:
        first = pose_result.get(first_name)
        second = pose_result.get(second_name)
        if first is None or second is None:
            continue

        first_point = tuple(int(round(value)) for value in first.position_2d)
        second_point = tuple(int(round(value)) for value in second.position_2d)
        cv2.line(frame, first_point, second_point, (0, 255, 255), 2)

    for keypoint in pose_result.keypoints.values():
        point = tuple(int(round(value)) for value in keypoint.position_2d)
        cv2.circle(frame, point, 4, (0, 0, 255), -1)
