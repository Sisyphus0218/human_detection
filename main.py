import hydra
from omegaconf import DictConfig

from target_detector import TargetDetector
from target_reid import TargetReID
from target_tracker import TargetTracker


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # STAGE 1: Build the target person's ReID gallery from reference images.

    # Detect the target person in the reference images and crop them.
    detector = TargetDetector(
        model_path=cfg.detector.model_path,
        confidence=cfg.detector.confidence,
        device=cfg.device,
    )

    detections = detector.detect_target(
        cfg.target.image_dir,
        cfg.target.crop_dir,
    )

    # Register the target person in the ReID gallery using the cropped images.
    reid = TargetReID(
        model_path=cfg.reid.model_path,
        threshold=cfg.reid.threshold,
        device=cfg.device,
        max_features=cfg.reid.max_features,
        top_k=cfg.reid.top_k,
    )

    reid.add_target_images([detection.crop for detection in detections])
    reid.register_target()
    print(f"Target gallery size: {reid.gallery_size}")

    # STAGE2: Find the registered target and track the target.
    tracker = TargetTracker(
        yolo_path=cfg.tracker.model_path,
        tracker_config=cfg.tracker.tracker_config,
        reid=reid,
        device=cfg.device,
        confidence=cfg.tracker.confidence,
        lost_tolerance=cfg.tracker.lost_tolerance,
        image_size=cfg.tracker.image_size,
    )

    tracker.track_video(
        input_path=cfg.video.input_path,
        output_path=cfg.video.output_path,
    )


if __name__ == "__main__":
    main()
