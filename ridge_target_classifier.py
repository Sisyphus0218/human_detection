import torch
from torch import nn


class RidgeTargetClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        regularization: float = 1.0,
        threshold: float = 0.65,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.regularization = regularization
        self.threshold = threshold
        self.device = torch.device(device)

        # The paper's ridge-regression formulation does not include a bias.
        self.register_buffer(
            "weight",
            torch.zeros(self.feature_dim, dtype=torch.float32, device=self.device),
        )

        self.is_trained = False

    def reset(self) -> None:
        self.weight.zero_()
        self.is_trained = False

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        features = features.to(device=self.device, dtype=torch.float32)
        return features @ self.weight

    @torch.inference_mode()
    def fit(
        self,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
    ) -> bool:
        """Fit the classifier from the supplied positive and negative features."""
        positive_features = positive_features.to(self.device)
        negative_features = negative_features.to(self.device)

        features = torch.cat(
            [positive_features, negative_features],
            dim=0,
        )

        positive_count = positive_features.size(0)
        negative_count = negative_features.size(0)

        labels = torch.cat(
            [
                torch.ones(
                    positive_count,
                    dtype=features.dtype,
                    device=self.device,
                ),
                torch.zeros(
                    negative_count,
                    dtype=features.dtype,
                    device=self.device,
                ),
            ],
            dim=0,
        )

        sample_count, feature_dim = features.shape
        if sample_count <= feature_dim:
            identity = torch.eye(
                sample_count,
                dtype=features.dtype,
                device=self.device,
            )
            dual_coefficients = torch.linalg.solve(
                features @ features.T + self.regularization * identity,
                labels,
            )
            fitted_weight = features.T @ dual_coefficients
        else:
            identity = torch.eye(
                feature_dim,
                dtype=features.dtype,
                device=self.device,
            )
            fitted_weight = torch.linalg.solve(
                features.T @ features + self.regularization * identity,
                features.T @ labels,
            )

        self.weight.copy_(fitted_weight)
        self.is_trained = True

        return True

    @torch.inference_mode()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw target scores; values are not restricted to [0, 1]."""
        if not self.is_trained:
            raise RuntimeError("Target classifier is not trained yet")

        self.eval()
        return self(features)

    @torch.inference_mode()
    def find_target(
        self,
        features: torch.Tensor,
    ) -> tuple[int | None, float, torch.Tensor]:
        """Return the highest-scoring candidate if it passes the threshold."""
        scores = self.predict(features)
        best_index = int(torch.argmax(scores).item())
        best_score = float(scores[best_index].item())
        if best_score < self.threshold:
            return None, best_score, scores.detach()

        return best_index, best_score, scores.detach()
