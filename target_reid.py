from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor


class TargetReID:
    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.65,
        device: str = "cpu",
        max_features: int = 30,
        top_k: int = 3,
    ):
        self.device = device
        self.threshold = threshold  # Minimum ReID similarity score required to accept a candidate as the target.
        self.max_features = max_features
        self.top_k = top_k

        self.feature_extractor = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=str(model_path),
            device=self.device,
            verbose=False,
        )

        self.target_images: list[np.ndarray] = []  # target bgr images
        self.target_features: torch.Tensor | None = None

    @property
    def gallery_size(self) -> int:
        """Return the number of features stored in the target gallery."""
        if self.target_features is None:
            return 0
        return self.target_features.shape[0]

    def add_target_images(
        self,
        target_images: np.ndarray | list[np.ndarray],
    ) -> None:
        # Append images to the target images gallery.
        if isinstance(target_images, np.ndarray):
            target_images = [target_images]
        self.target_images.extend(target_images)

    @torch.inference_mode()
    def extract_features(
        self,
        bgr_images: np.ndarray | list[np.ndarray],
    ) -> torch.Tensor:
        if isinstance(bgr_images, np.ndarray):
            bgr_images = [bgr_images]

        rgb_images = []
        for index, image in enumerate(bgr_images):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_image)

        features = self.feature_extractor(rgb_images)
        return F.normalize(features, p=2, dim=1)

    def register_target(self) -> torch.Tensor:
        features = self.extract_features(self.target_images)
        self.target_features = features[-self.max_features :].detach().clone()
        return self.target_features

    def clear_gallery(self) -> None:
        self.target_images = []
        self.target_features = None

    def calculate_scores(
        self,
        candidate_images: np.ndarray | list[np.ndarray],
    ) -> np.ndarray:
        # Calculate cosine similarity between candidates and the target gallery.
        if self.target_features is None:
            raise RuntimeError("Register the target before calculating scores")

        candidate_features = self.extract_features(candidate_images)
        similarity_matrix = candidate_features @ self.target_features.T

        number_of_neighbors = min(self.top_k, self.gallery_size)
        scores = similarity_matrix.topk(
            k=number_of_neighbors,
            dim=1,
        ).values.mean(dim=1)

        return scores.detach().cpu().numpy()

    def find_target(
        self,
        candidate_images: list[np.ndarray],
    ) -> tuple[int | None, float, np.ndarray]:
        # Find the candidate most similar to the registered target.
        candidate_images = list(candidate_images)
        if not candidate_images:
            return None, 0.0, np.empty(0, dtype=np.float32)

        scores = self.calculate_scores(candidate_images)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        # If the best candidate's score is below the threshold, reject it and return None.
        if best_score < self.threshold:
            return None, best_score, scores

        return best_index, best_score, scores
