"""Feature extractor interfaces and implementations."""

from .feature_extractor import FeatureExtractor
from .osnet_feature_extractor import OSNetFeatureExtractor

__all__ = ["FeatureExtractor", "OSNetFeatureExtractor"]
