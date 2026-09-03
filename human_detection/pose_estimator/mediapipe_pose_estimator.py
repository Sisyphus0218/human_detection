from pathlib import Path

import cv2
import numpy as np

from .pose_estimation_result import PoseEstimationResult, PoseKeypoint
from .pose_estimator import PoseEstimator

MEDIAPIPE_POSE_KEYPOINT_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


class MediaPipePoseEstimator(PoseEstimator):
    """Estimate 2D pose keypoints in a tracked-person crop with MediaPipe."""

    def __init__(
        self,
        model_path: str | Path,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_keypoint_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe is not installed. Install the 'mediapipe' package "
                "to use MediaPipePoseEstimator."
            ) from error

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model not found: {model_path}")

        if not 0.0 <= min_keypoint_confidence <= 1.0:
            raise ValueError("min_keypoint_confidence must be between 0 and 1")

        self._mp = mp
        self.min_keypoint_confidence = min_keypoint_confidence

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def estimate(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> PoseEstimationResult | None:
        if bbox is None:
            return None

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3)")

        frame_height, frame_width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, frame_width))
        x2 = int(np.clip(x2, 0, frame_width))
        y1 = int(np.clip(y1, 0, frame_height))
        y2 = int(np.clip(y2, 0, frame_height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(crop_rgb),
        )
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        crop_height, crop_width = crop_bgr.shape[:2]
        landmarks = result.pose_landmarks[0]
        keypoints: dict[str, PoseKeypoint] = {}

        for name, landmark in zip(MEDIAPIPE_POSE_KEYPOINT_NAMES, landmarks):
            visibility = float(landmark.visibility or 0.0)
            presence = float(landmark.presence or 0.0)
            confidence = min(visibility, presence)
            if confidence < self.min_keypoint_confidence:
                continue

            u = x1 + float(landmark.x) * crop_width
            v = y1 + float(landmark.y) * crop_height
            u = float(np.clip(u, 0, frame_width - 1))
            v = float(np.clip(v, 0, frame_height - 1))

            keypoints[name] = PoseKeypoint(
                name=name,
                position_2d=(u, v),
                confidence=confidence,
            )

        if not keypoints:
            return None

        return PoseEstimationResult(keypoints=keypoints)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
