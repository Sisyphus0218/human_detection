from abc import ABC, abstractmethod

import torch
from torch import nn


class TargetClassifier(nn.Module, ABC):
    """Common interface for target classifiers."""

    is_trained: bool

    @abstractmethod
    def reset(self) -> None:
        """Reset the classifier to its untrained state."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate classification scores for feature rows."""

    @abstractmethod
    def fit(
        self,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
    ) -> bool:
        """Fit the classifier using positive and negative features."""

    @abstractmethod
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict target scores for feature rows."""

    @abstractmethod
    def find_target(
        self,
        features: torch.Tensor,
    ) -> tuple[int | None, float, torch.Tensor]:
        """Return the best candidate when its score passes the threshold."""
