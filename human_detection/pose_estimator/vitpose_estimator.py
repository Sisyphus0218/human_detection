from pathlib import Path

import numpy as np

from .pose_estimation_result import PoseEstimationResult, PoseKeypoint
from .pose_estimator import PoseEstimator


COCO_POSE_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class ViTPoseEstimator(PoseEstimator):
    """Estimate one tracked person's 2D pose with top-down ViTPose."""

    def __init__(
        self,
        model_id: str = "usyd-community/vitpose-base-simple",
        cache_dir: str | Path | None = None,
        device: str | None = None,
        min_keypoint_confidence: float = 0.3,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, VitPoseForPoseEstimation
        except ImportError as error:
            raise RuntimeError(
                "ViTPose dependencies are not installed. Install 'transformers' "
                "to use ViTPoseEstimator."
            ) from error

        if not 0.0 <= min_keypoint_confidence <= 1.0:
            raise ValueError("min_keypoint_confidence must be between 0 and 1")

        self._torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_keypoint_confidence = min_keypoint_confidence

        resolved_cache_dir = None
        if cache_dir is not None:
            resolved_cache_dir = Path(cache_dir)
            resolved_cache_dir.mkdir(parents=True, exist_ok=True)

        self._processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=resolved_cache_dir,
        )
        self._model = VitPoseForPoseEstimation.from_pretrained(
            model_id,
            cache_dir=resolved_cache_dir,
        )
        self._model.to(self.device)
        self._model.eval()

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

        # ViTPose expects RGB images and COCO boxes: (x, y, width, height).
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        person_boxes = np.asarray(
            [[x1, y1, x2 - x1, y2 - y1]],
            dtype=np.float32,
        )

        inputs = self._processor(
            images=frame_rgb,
            boxes=[person_boxes],
            return_tensors="pt",
        ).to(self.device)

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_pose_estimation(
            outputs,
            boxes=[person_boxes],
        )
        if not results or not results[0]:
            return None

        person_result = results[0][0]
        positions = person_result["keypoints"].detach().cpu().numpy()
        scores = person_result["scores"].detach().cpu().numpy()

        keypoints: dict[str, PoseKeypoint] = {}
        for name, position, score in zip(
            COCO_POSE_KEYPOINT_NAMES,
            positions,
            scores,
        ):
            confidence = float(score)
            if confidence < self.min_keypoint_confidence:
                continue

            keypoints[name] = PoseKeypoint(
                name=name,
                position_2d=(float(position[0]), float(position[1])),
                confidence=confidence,
            )

        if not keypoints:
            return None

        return PoseEstimationResult(keypoints=keypoints)
