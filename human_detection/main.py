import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from human_detection.tracking_runner import TrackingRunner
from human_detection.visualization import VideoWriter


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # STAGE 1: Register the target person.
    # Detect the target person in the reference images and crop bboxes.
    # Register the target person and save to the feature memory.
    person_detector = instantiate(cfg.person_detector)
    crops = person_detector.detect_target(
        cfg.target.directory,
        cfg.results.crops.path,
    )

    feature_extractor = instantiate(cfg.feature_extractor)

    positive_feature_memory = instantiate(cfg.positive_memory)
    negative_feature_memory = instantiate(cfg.negative_memory)

    positive_features = feature_extractor.extract_features(crops)
    positive_feature_memory.add_long_term_memory(positive_features)

    print(f"Register successful.")

    # STAGE 2: Track the registered target.
    person_tracker = instantiate(cfg.person_tracker)
    target_gallery_matcher = instantiate(cfg.target_gallery_matcher)
    target_classifier = instantiate(cfg.target_classifier)
    target_bbox_trajectory = instantiate(cfg.target_bbox_trajectory)
    frame_source = instantiate(cfg.input.source)

    target_tracker = instantiate(
        cfg.target_tracker,
        person_tracker=person_tracker,
        feature_extractor=feature_extractor,
        positive_feature_memory=positive_feature_memory,
        negative_feature_memory=negative_feature_memory,
        target_gallery_matcher=target_gallery_matcher,
        target_classifier=target_classifier,
        target_bbox_trajectory=target_bbox_trajectory,
    )

    tracking_writer = None
    if cfg.outputs.tracking_video.enabled:
        tracking_writer = VideoWriter(
            output_path=cfg.outputs.tracking_video.path,
            fps=frame_source.fps,
        )

    debug_writer = None
    if cfg.outputs.debug_video.enabled:
        debug_writer = VideoWriter(
            output_path=cfg.outputs.debug_video.path,
            fps=frame_source.fps,
        )

    runner = TrackingRunner(target_tracker=target_tracker)
    runner.run(
        source=frame_source,
        tracking_writer=tracking_writer,
        debug_writer=debug_writer,
        bbox_output_path=(cfg.outputs.bbox.path if cfg.outputs.bbox.enabled else None),
        display_enabled=cfg.display.enabled,
        display_window_name=cfg.display.window_name,
    )


if __name__ == "__main__":
    main()
