import torch

from feature_memory import FeatureMemory


class TargetGalleryMatcher:
    def __init__(
        self,
        threshold: float = 0.65,
        top_k: int = 3,
        device: str = "cpu",
    ):
        self.device = device
        self.threshold = threshold
        self.top_k = top_k  # Number of nearest gallery features used for scoring.

    def calculate_similarity_scores(
        self,
        query_features: torch.Tensor,
        target_gallery: FeatureMemory,
    ) -> torch.Tensor:
        """Calculate each query's similarity score against the target gallery."""
        gallery_features = target_gallery.all_features()

        # [num_queries, feature_dim] @ [num_gallery, feature_dim].T
        # -> [num_queries, num_gallery]
        #            gallery_feature1  gallery_feature2 ...
        # query1     similarity1_1    similarity1_2
        # query2     similarity2_1    similarity2_2
        # ...
        similarity_matrix = query_features @ gallery_features.T

        # Average each query's top-k similarities to the target gallery.
        num_neighbors = min(self.top_k, gallery_features.shape[0])
        similarity_scores = similarity_matrix.topk(
            k=num_neighbors,
            dim=1,
        ).values.mean(dim=1)

        return similarity_scores.detach()

    def find_target(
        self,
        query_features: torch.Tensor,
        target_gallery: FeatureMemory,
    ) -> tuple[int | None, float, torch.Tensor]:
        """Find the query with the highest target-gallery similarity score."""
        scores = self.calculate_similarity_scores(
            query_features,
            target_gallery,
        )
        best_index = int(torch.argmax(scores).item())
        best_score = float(scores[best_index].item())

        # Reject the best query when it does not match the target gallery strongly enough.
        if best_score < self.threshold:
            return None, best_score, scores

        return best_index, best_score, scores
