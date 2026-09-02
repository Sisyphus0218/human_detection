from dataclasses import dataclass
from pathlib import Path

from human_detection.feature_extractor import FeatureExtractor
from human_detection.person_detector import PersonDetector
from human_detection.target_matching import FeatureMemory


@dataclass(frozen=True)
class RegistrationPipelineConfig:
    """Input and output paths for target registration."""

    target_directory: str | Path
    crops_output_path: str | Path


class RegistrationPipeline:
    """Register a target person from reference images."""

    def __init__(
        self,
        person_detector: PersonDetector,
        feature_extractor: FeatureExtractor,
        feature_memory: FeatureMemory,
        config: RegistrationPipelineConfig,
    ) -> None:
        self.person_detector = person_detector
        self.feature_extractor = feature_extractor
        self.feature_memory = feature_memory
        self.config = config

    def run(self) -> None:
        crops = self.person_detector.detect_target(
            self.config.target_directory,
            self.config.crops_output_path,
        )
        features = self.feature_extractor.extract_features(crops)

        self.feature_memory.clear()
        self.feature_memory.add_long_term_memory(features)

        print("Registration successful.")
