import numpy as np

from human_detection.pose_estimator.pose_estimator import PoseEstimationResult
from .position_estimation_result import PositionEstimationResult


class PositionEstimator:
    def __init__(self, intrinsics):
        # intrinsics = [[fx, 0, cx],
        #               [0, fy, cy],
        #               [0, 0, 1]]
        if intrinsics is None:
            self.intrinsics = None
        else:
            self.intrinsics = np.asarray(intrinsics, dtype=np.float32)

    @staticmethod
    def estimate_intrinsics(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise ValueError("frame must be a NumPy array with at least 2 dimensions")

        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")

        focal_length = float(np.hypot(width, height))
        return np.asarray(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def deproject(
        self,
        depth_mm: float,
        pixel: tuple[float, float],
    ) -> tuple[float, float, float]:
        """
        Deproject a pixel coordinate (u, v) with depth value (Z) to 3D coordinates (X, Y, Z).

        Args:
            depth_mm: Depth value in millimeters.
            pixel: Pixel coordinates (u, v).

        Returns:
            (X, Y, Z): 3D coordinates in millimeters.
        """

        u, v = pixel

        fx = float(self.intrinsics[0, 0])
        fy = float(self.intrinsics[1, 1])
        cx = float(self.intrinsics[0, 2])
        cy = float(self.intrinsics[1, 2])

        Z = float(depth_mm)

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return X, Y, Z

    def estimate_keypoint_depth(
        self,
        depth_mm: np.ndarray,
        pixel: tuple[float, float],
        radius: int = 5,
    ) -> float | None:
        u, v = pixel
        height, width = depth_mm.shape[:2]

        u = int(round(u))
        v = int(round(v))
        if not (0 <= u < width and 0 <= v < height):
            return None

        x1 = max(0, u - radius)
        x2 = min(width, u + radius + 1)
        y1 = max(0, v - radius)
        y2 = min(height, v + radius + 1)

        roi_depth = depth_mm[y1:y2, x1:x2]
        valid_depth = roi_depth[
            np.isfinite(roi_depth) & (roi_depth > 10) & (roi_depth < 10_000)
        ]  # 1 cm～10 m

        if valid_depth.size == 0:
            return None

        depth = float(np.median(valid_depth))

        return depth

    def estimate(
        self,
        depth_mm: np.ndarray | None,
        pose: PoseEstimationResult | None,
    ) -> PositionEstimationResult | None:
        if depth_mm is None or pose is None:
            return None

        if self.intrinsics is None:
            self.intrinsics = self.estimate_intrinsics(depth_mm)

        joint_pairs = {
            "neck": ("left_shoulder", "right_shoulder"),
            "hip": ("left_hip", "right_hip"),
            "knee": ("left_knee", "right_knee"),
            "ankle": ("left_ankle", "right_ankle"),
        }

        positions_3d = {}

        for center_name, (left_name, right_name) in joint_pairs.items():
            left_keypoint = pose.get(left_name)
            right_keypoint = pose.get(right_name)

            if left_keypoint is None or right_keypoint is None:
                continue

            left_depth = self.estimate_keypoint_depth(
                depth_mm=depth_mm,
                pixel=left_keypoint.position_2d,
            )
            right_depth = self.estimate_keypoint_depth(
                depth_mm=depth_mm,
                pixel=right_keypoint.position_2d,
            )

            if left_depth is None or right_depth is None:
                continue

            left_position_3d = self.deproject(
                depth_mm=left_depth,
                pixel=left_keypoint.position_2d,
            )
            right_position_3d = self.deproject(
                depth_mm=right_depth,
                pixel=right_keypoint.position_2d,
            )

            center_position_3d = tuple(
                (
                    (left + right) / 2
                    for left, right in zip(left_position_3d, right_position_3d)
                )
            )

            positions_3d[center_name] = center_position_3d

        if not positions_3d:
            return None

        return PositionEstimationResult(**positions_3d)
