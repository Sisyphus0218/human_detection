import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # STAGE 1: Register the target person.
    person_detector = instantiate(cfg.person_detector)
    feature_extractor = instantiate(cfg.feature_extractor)
    positive_feature_memory = instantiate(cfg.target_matching.positive_memory)
    negative_feature_memory = instantiate(cfg.target_matching.negative_memory)

    registration = instantiate(
        cfg.registration_pipeline,
        person_detector=person_detector,
        feature_extractor=feature_extractor,
        feature_memory=positive_feature_memory,
    )
    registration.run()

    # STAGE 2: Track the registered target.
    person_tracker = instantiate(cfg.person_tracker)
    target_gallery_matcher = instantiate(cfg.target_matching.target_gallery_matcher)
    target_classifier = instantiate(cfg.target_classifier)
    target_bbox_trajectory = instantiate(
        cfg.target_tracker.target_bbox_trajectory,
    )
    frame_source = instantiate(cfg.source)

    target_tracker = instantiate(
        cfg.target_tracker.tracker,
        person_tracker=person_tracker,
        feature_extractor=feature_extractor,
        positive_feature_memory=positive_feature_memory,
        negative_feature_memory=negative_feature_memory,
        target_gallery_matcher=target_gallery_matcher,
        target_classifier=target_classifier,
        target_bbox_trajectory=target_bbox_trajectory,
    )

    tracking = instantiate(
        cfg.tracking_pipeline,
        target_tracker=target_tracker,
    )
    tracking.run(frame_source)


if __name__ == "__main__":
    main()
