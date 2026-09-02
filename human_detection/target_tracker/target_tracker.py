from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from human_detection.feature_extractor import FeatureExtractor
from human_detection.frame_source import FrameSource
from human_detection.person_tracker import (
    PersonTracker,
    PersonTrackingResult,
    TrackedPerson,
)
from human_detection.target_classifier import TargetClassifier
from human_detection.target_matching import FeatureMemory, TargetGalleryMatcher
from human_detection.utils import calculate_iou

from .target_bbox_trajectory import TargetBBoxTrajectory
from .target_tracking_result import TargetTrackingResult


@dataclass(frozen=True)
class TargetTrackerConfig:
    reid_keep_threshold: float
    reid_reject_threshold: float
    reid_validation_interval: int
    reid_validation_tolerance: int
    missing_frame_tolerance: int
    classifier_collect_interval: int
    classifier_max_overlap_iou: float
    min_negatives: int
    min_positives: int


class TargetTracker:
    def __init__(
        self,
        person_tracker: PersonTracker,
        feature_extractor: FeatureExtractor,
        positive_feature_memory: FeatureMemory,
        negative_feature_memory: FeatureMemory,
        target_gallery_matcher: TargetGalleryMatcher,
        target_classifier: TargetClassifier,
        target_bbox_trajectory: TargetBBoxTrajectory,
        config: TargetTrackerConfig,
    ) -> None:
        self.person_tracker = person_tracker
        self.feature_extractor = feature_extractor
        self.positive_feature_memory = positive_feature_memory
        self.negative_feature_memory = negative_feature_memory
        self.target_gallery_matcher = target_gallery_matcher
        self.target_classifier = target_classifier
        self.target_bbox_trajectory = target_bbox_trajectory

        # ReID state-machine parameters.
        self.state = "SEARCHING"
        self.target_track_id: int | None = None

        self.reid_keep_threshold = config.reid_keep_threshold
        self.reid_reject_threshold = config.reid_reject_threshold

        self.reid_validation_interval = config.reid_validation_interval
        self.reid_validation_tolerance = config.reid_validation_tolerance
        self.reid_validation_failures = 0

        self.missing_frame_tolerance = config.missing_frame_tolerance
        self.missing_frames = 0

        # target classifier
        self.classifier_collect_interval = config.classifier_collect_interval
        self.classifier_max_overlap_iou = config.classifier_max_overlap_iou
        self.min_negatives = config.min_negatives
        self.min_positives = config.min_positives

    @staticmethod
    def crop_person(frame: np.ndarray, person: TrackedPerson) -> np.ndarray:
        """Copy one tracked person's BGR crop from the current frame."""
        x1, y1, x2, y2 = person.bbox
        return frame[y1:y2, x1:x2].copy()

    def update_feature_memory(
        self,
        frame: np.ndarray,
        other_persons: list[TrackedPerson],
        target: TrackedPerson,
        target_features: torch.Tensor,
    ) -> None:
        is_target_clean = True
        safe_negative_crops = []

        for person in other_persons:
            iou = calculate_iou(target.bbox, person.bbox)
            if iou <= self.classifier_max_overlap_iou:
                safe_negative_crops.append(self.crop_person(frame, person))
            else:
                is_target_clean = False

        if safe_negative_crops:
            negative_features = self.feature_extractor.extract_features(
                safe_negative_crops
            )
            self.negative_feature_memory.add_short_term_memory(negative_features)

        if is_target_clean:
            self.positive_feature_memory.add_short_term_memory(target_features)

    def update_target_classifier(
        self,
        frame_index: int,
    ) -> None:
        if frame_index % self.classifier_collect_interval != 0:
            return

        positive_count = self.positive_feature_memory.count
        negative_count = self.negative_feature_memory.count
        if positive_count < self.min_positives or negative_count < self.min_negatives:
            tqdm.write(
                f"Frame {frame_index}: not enough samples to update target classifier, "
                f"positives={positive_count}, negatives={negative_count}"
            )
            return

        self.target_classifier.fit(
            positive_features=self.positive_feature_memory.all_features(),
            negative_features=self.negative_feature_memory.all_features(),
        )

        tqdm.write(f"Frame {frame_index}: target classifier updated")

    def search_target(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        if not tracked_persons:
            return None

        crops = [self.crop_person(frame, person) for person in tracked_persons]
        features = self.feature_extractor.extract_features(crops)

        if self.target_classifier.is_trained:
            best_index, best_score, _ = self.target_classifier.find_target(
                features.detach()
            )
        else:
            best_index, best_score, _ = self.target_gallery_matcher.find_target(
                features.detach(), self.positive_feature_memory
            )

        if best_index is None:
            tqdm.write(
                f"Frame {frame_index}: no candidate passed the ReID threshold, "
                f"best score: {best_score:.4f}"
            )
            return None

        target = tracked_persons[best_index]

        self.state = "TRACKING"
        self.target_track_id = target.track_id
        self.reid_validation_failures = 0

        tqdm.write(
            f"Frame {frame_index}: target confirmed, "
            f"track id={self.target_track_id}, "
            f"reid score={best_score:.4f}"
        )

        return target

    def track_target(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """Follow the locked ID and periodically validate its appearance."""
        # search the locked ID in the current frame
        target = None
        for index, person in enumerate(tracked_persons):
            if person.track_id == self.target_track_id:
                target = person
                target_index = index
                other_persons = (
                    tracked_persons[:target_index] + tracked_persons[target_index + 1 :]
                )
                break

        # the target disappears
        if target is None:
            tqdm.write(
                f"Frame {frame_index}: target ID missing, " f"switch to RECOVERING"
            )
            self.state = "RECOVERING"
            self.reid_validation_failures = 0
            return self.recover_target(
                frame=frame,
                tracked_persons=tracked_persons,
                frame_index=frame_index,
            )

        # validate the target's appearance every N frames
        if frame_index % self.reid_validation_interval != 0:
            return target
        else:  # validate
            # calculate reid score
            target_crop = self.crop_person(frame, target)
            target_features = self.feature_extractor.extract_features(target_crop)
            if self.target_classifier.is_trained:
                score = self.target_classifier.predict(target_features.detach())[0]
            else:
                score = self.target_gallery_matcher.calculate_similarity_scores(
                    target_features.detach(), self.positive_feature_memory
                )[0]

            # validate target
            if score >= self.reid_keep_threshold:  # pass
                self.reid_validation_failures = 0
                self.update_feature_memory(
                    frame=frame,
                    other_persons=other_persons,
                    target=target,
                    target_features=target_features.detach(),
                )
                self.update_target_classifier(
                    frame_index=frame_index,
                )
            elif score < self.reid_reject_threshold:  # fail
                self.reid_validation_failures = self.reid_validation_tolerance
            else:  # supcious
                self.reid_validation_failures += 1

            if self.reid_validation_failures < self.reid_validation_tolerance:
                return target
            else:  # too many failures, switch to SEARCHING
                tqdm.write(
                    f"Frame {frame_index}: locked ID failed ReID validation, "
                    f"track id={target.track_id}, "
                    f"ReID score={score:.4f}"
                )
                self.state = "SEARCHING"
                self.target_track_id = None
                self.reid_validation_failures = 0
                return self.search_target(
                    frame=frame,
                    tracked_persons=tracked_persons,
                    frame_index=frame_index,
                )

    def recover_target(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        if self.missing_frames >= self.missing_frame_tolerance:
            self.state = "SEARCHING"
            self.target_track_id = None
            self.missing_frames = 0
            return self.search_target(
                frame=frame,
                tracked_persons=tracked_persons,
                frame_index=frame_index,
            )

        if not tracked_persons:
            self.missing_frames += 1
            return None

        for person in tracked_persons:
            if person.track_id == self.target_track_id:
                self.state = "TRACKING"
                self.missing_frames = 0
                tqdm.write(
                    f"Frame {frame_index}: recovered original track ID, "
                    f"track id={person.track_id}"
                )
                return person

        crops = [self.crop_person(frame, person) for person in tracked_persons]
        features = self.feature_extractor.extract_features(crops)
        if self.target_classifier.is_trained:
            recovery_method = "target classifier"
            best_index, best_score, _ = self.target_classifier.find_target(
                features.detach()
            )
        else:
            recovery_method = "gallery matcher"
            best_index, best_score, _ = self.target_gallery_matcher.find_target(
                features.detach(), self.positive_feature_memory
            )
        if best_index is not None:
            target = tracked_persons[best_index]
            self.missing_frames = 0
            self.target_track_id = target.track_id
            self.state = "TRACKING"
            tqdm.write(
                f"Frame {frame_index}: recovered target with {recovery_method}, "
                f"track id={target.track_id}, ReID score={best_score:.4f}"
            )
            return target

        self.missing_frames += 1

        return None

    def update_target_state(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """
        SEARCHING locks the best qualified global ReID candidate.
        TRACKING follows the locked ID and periodically validates appearance.
        """
        if self.state == "SEARCHING":
            return self.search_target(frame, tracked_persons, frame_index)

        if self.state == "TRACKING":
            return self.track_target(frame, tracked_persons, frame_index)

        if self.state == "RECOVERING":
            return self.recover_target(frame, tracked_persons, frame_index)

        raise RuntimeError(f"Unknown tracking state: {self.state}")

    def process_frame(
        self,
        frame_index: int,
        tracking_result: PersonTrackingResult,
    ) -> TargetTrackingResult:
        """Update target state for one person-tracking result."""
        frame_bgr = tracking_result.frame
        target = self.update_target_state(
            frame=frame_bgr,
            tracked_persons=tracking_result.tracked_persons,
            frame_index=frame_index,
        )

        frame_height, frame_width = frame_bgr.shape[:2]
        trajectory_entry = self.target_bbox_trajectory.update(
            frame_index=frame_index,
            observed_target=target,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        return TargetTrackingResult(
            frame_index=frame_index,
            person_tracking_result=tracking_result,
            target=target,
            trajectory_entry=trajectory_entry,
            state=self.state,
        )

    def reset(self) -> None:
        """Reset state for a new tracking source."""
        self.state = "SEARCHING"
        self.target_track_id = None
        self.missing_frames = 0
        self.target_bbox_trajectory.reset()
        self.reid_validation_failures = 0
        self.target_classifier.reset()

    def track(
        self,
        source: FrameSource,
    ) -> Iterator[TargetTrackingResult]:
        """Yield one target-tracking result for each source frame."""
        self.reset()

        with source:
            while True:
                rgbd_frame = source.read()
                if rgbd_frame is None:
                    break

                tracking_result = self.person_tracker.track_frame(rgbd_frame.color_bgr)
                target_result = self.process_frame(
                    frame_index=rgbd_frame.frame_index,
                    tracking_result=tracking_result,
                )

                yield target_result
