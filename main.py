import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # STAGE 1: Detect the target person in the reference images and crop bboxes.
    detector = instantiate(cfg.detector)

    crops = detector.detect_target(
        cfg.target.image_dir,
        cfg.target.crop_dir,
    )

    # STAGE 2: Register the target person and save to the feature memory.
    person_feature_extractor = instantiate(cfg.person_feature_extractor)

    positive_feature_memory = instantiate(cfg.positive_memory)
    negative_feature_memory = instantiate(cfg.negative_memory)

    positive_features = person_feature_extractor.extract_features(crops)
    positive_feature_memory.add_long_term_memory(positive_features)
    print(f"Register successful.")

    # STAGE 3: Track the registered target in the video.
    target_gallery_matcher = instantiate(cfg.target_gallery_matcher)

    person_tracker = instantiate(cfg.person_tracker)

    target_classifier = instantiate(cfg.target_classifier)

    tracking_pipeline = instantiate(
        cfg.tracking_pipeline,
        tracker=person_tracker,
        person_feature_extractor=person_feature_extractor,
        positive_feature_memory=positive_feature_memory,
        negative_feature_memory=negative_feature_memory,
        target_gallery_matcher=target_gallery_matcher,
        target_classifier=target_classifier,
    )

    tracking_pipeline.track_video(
        input_path=cfg.video.input_path,
        output_path=cfg.video.output_path,
        bbox_enabled=cfg.bbox.enabled,
        bbox_output_path=cfg.bbox.path,
    )


if __name__ == "__main__":
    main()
