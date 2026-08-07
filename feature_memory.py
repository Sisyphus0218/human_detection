from collections import deque

import torch


class FeatureMemory:
    def __init__(
        self,
        feature_dim: int,
        short_term_capacity: int,
        long_term_capacity: int,
        device: str = "cpu",
    ):
        self.feature_dim = feature_dim
        self.device = torch.device(device)

        self.short_term_memory: deque[torch.Tensor] = deque(maxlen=short_term_capacity)
        self.long_term_memory: deque[torch.Tensor] = deque(maxlen=long_term_capacity)

    @property
    def short_term_count(self) -> int:
        return len(self.short_term_memory)

    @property
    def long_term_count(self) -> int:
        return len(self.long_term_memory)

    @property
    def count(self) -> int:
        return len(self.short_term_memory) + len(self.long_term_memory)

    def add_short_term_memory(
        self,
        features: torch.Tensor | None,
    ) -> None:
        if features is None:
            return

        for feature in features:
            self.short_term_memory.append(feature.detach().clone())

    def add_long_term_memory(
        self,
        features: torch.Tensor | None,
    ) -> None:
        if features is None:
            return

        for feature in features:
            self.long_term_memory.append(feature.detach().clone())

    def clear_short_term_memory(self) -> None:
        self.short_term_memory.clear()

    def clear_long_term_memory(self) -> None:
        self.long_term_memory.clear()

    def clear(self) -> None:
        self.short_term_memory.clear()
        self.long_term_memory.clear()

    def all_features(
        self,
    ) -> torch.Tensor:
        features = [
            *self.long_term_memory,
            *self.short_term_memory,
        ]

        if not features:
            return torch.empty(
                (0, self.feature_dim),
                dtype=torch.float32,
                device=self.device,
            )

        return torch.stack(
            features,
            dim=0,
        ).to(device=self.device)
