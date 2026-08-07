import torch
from torch import nn


class TargetClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        update_steps: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold: float = 0.65,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.update_steps = update_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.device = torch.device(device)

        self.classifier = nn.Linear(
            in_features=self.feature_dim,
            out_features=1,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.is_trained = False
        self.last_loss: float | None = None

    def reset(self) -> None:
        # Start the classifier from new random parameters.
        self.classifier.reset_parameters()
        self.classifier.to(self.device)

        # Clear AdamW momentum and other optimizer state.
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.is_trained = False
        self.last_loss = None

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate unnormalized classification scores.

        Args:
            features: ReID features with shape [batch_size, feature_dim].

        Returns:
            Logits with shape [batch_size].
        """
        features = features.to(
            device=self.device,
            dtype=torch.float32,
        )

        if features.ndim != 2:
            raise ValueError(
                "Classifier input must have shape " "[batch_size, feature_dim]"
            )

        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {features.shape[1]}"
            )

        return self.classifier(features).squeeze(1)

    def fit(
        self,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
    ) -> bool:
        # features and labels
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
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.zeros(
                    negative_count,
                    dtype=torch.float32,
                    device=self.device,
                ),
            ],
            dim=0,
        )

        # loss function
        positive_weight = torch.tensor(
            [negative_count / positive_count],
            dtype=torch.float32,
            device=self.device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight,  # Give positive samples more weight when negatives are more numerous.
        )

        self.train()

        for _ in range(self.update_steps):
            logits = self(features)
            loss = criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.eval()

        self.last_loss = float(loss.detach().item())
        self.is_trained = True

        return True

    @torch.inference_mode()
    def predict(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return target probabilities for ReID feature rows.

        Args:
            features: Tensor with shape [batch_size, feature_dim] or [feature_dim].

        Returns:
            Probabilities with shape [batch_size].
        """
        if not self.is_trained:
            raise RuntimeError("Target classifier is not trained yet")

        self.eval()
        logits = self(features)
        return torch.sigmoid(logits)

    @torch.inference_mode()
    def find_target(
        self,
        features: torch.Tensor,
    ) -> tuple[int | None, float, torch.Tensor]:
        scores = self.predict(features)
        best_index = int(torch.argmax(scores).item())
        best_score = float(scores[best_index].item())
        if best_score < self.threshold:
            return None, best_score, scores.detach()

        return best_index, best_score, scores.detach()
