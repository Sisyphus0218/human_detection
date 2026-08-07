import hydra
from omegaconf import DictConfig

from feature_memory import FeatureMemory
from person_feature_extractor import PersonFeatureExtractor
from person_tracker import PersonTracker
from target_classifier import TargetClassifier
from target_detector import TargetDetector
from target_gallery_matcher import TargetGalleryMatcher
from target_tracking_pipeline import TargetTrackingPipeline


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # STAGE 1: Detect the target person in the reference images and crop bboxes.
    detector = TargetDetector(
        model_path=cfg.detector.model_path,
        confidence=cfg.detector.confidence,
        device=cfg.device,
    )

    crops = detector.detect_target(
        cfg.target.image_dir,
        cfg.target.crop_dir,
    )

    # STAGE 2: Register the target person and save to the feature memory.
    person_feature_extractor = PersonFeatureExtractor(
        model_name=cfg.person_feature_extractor.model_name,
        model_path=cfg.person_feature_extractor.model_path,
        device=cfg.device,
    )

    positive_feature_memory = FeatureMemory(
        feature_dim=cfg.positive_memory.feature_dim,
        short_term_capacity=cfg.positive_memory.short_term_capacity,
        long_term_capacity=cfg.positive_memory.long_term_capacity,
        device=cfg.device,
    )
    negative_feature_memory = FeatureMemory(
        feature_dim=cfg.negative_memory.feature_dim,
        short_term_capacity=cfg.negative_memory.short_term_capacity,
        long_term_capacity=cfg.negative_memory.long_term_capacity,
        device=cfg.device,
    )

    positive_features = person_feature_extractor.extract_features(crops)
    positive_feature_memory.add_long_term_memory(positive_features)
    print(f"Register successful.")

    # STAGE 3: Track the registered target in the video.
    target_gallery_matcher = TargetGalleryMatcher(
        threshold=cfg.target_gallery_matcher.threshold,
        top_k=cfg.target_gallery_matcher.top_k,
        device=cfg.device,
    )

    person_tracker = PersonTracker(
        yolo_path=cfg.person_tracker.model_path,
        tracker_config=cfg.person_tracker.tracker_config,
        confidence=cfg.person_tracker.confidence,
        image_size=cfg.person_tracker.image_size,
        device=cfg.device,
    )

    target_classifier = TargetClassifier(
        feature_dim=cfg.target_classifier.feature_dim,
        update_steps=cfg.target_classifier.update_steps,
        learning_rate=cfg.target_classifier.learning_rate,
        weight_decay=cfg.target_classifier.weight_decay,
        threshold=cfg.target_classifier.threshold,
        device=cfg.device,
    )

    tracking_pipeline = TargetTrackingPipeline(
        tracker=person_tracker,
        person_feature_extractor=person_feature_extractor,
        positive_feature_memory=positive_feature_memory,
        negative_feature_memory=negative_feature_memory,
        target_gallery_matcher=target_gallery_matcher,
        target_classifier=target_classifier,
        lost_tolerance=cfg.tracking_pipeline.lost_tolerance,
    )

    tracking_pipeline.track_video(
        input_path=cfg.video.input_path,
        output_path=cfg.video.output_path,
        bbox_enabled=cfg.bbox.enabled,
        bbox_output_path=cfg.bbox.path,
    )


if __name__ == "__main__":
    main()
