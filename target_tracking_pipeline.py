import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from target_classifier import TargetClassifier
from person_feature_extractor import PersonFeatureExtractor
from feature_memory import FeatureMemory
from person_tracker import PersonTracker
from target_gallery_matcher import TargetGalleryMatcher


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    crop: np.ndarray


class TargetTrackingPipeline:
    def __init__(
        self,
        tracker: PersonTracker,
        person_feature_extractor: PersonFeatureExtractor,
        positive_feature_memory: FeatureMemory,
        negative_feature_memory: FeatureMemory,
        target_gallery_matcher: TargetGalleryMatcher,
        target_classifier: TargetClassifier,
        lost_tolerance: int = 5,
    ) -> None:
        self.tracker = tracker
        self.person_feature_extractor = person_feature_extractor
        self.positive_feature_memory = positive_feature_memory
        self.negative_feature_memory = negative_feature_memory
        self.target_gallery_matcher = target_gallery_matcher
        self.target_classifier = target_classifier

        # ReID state-machine parameters.
        self.state = "SEARCHING"
        self.target_track_id: int | None = None

        self.reid_keep_threshold = 0.35
        self.reid_reject_threshold = 0.25

        self.reid_validation_interval = 5
        self.reid_validation_failures = 0
        self.reid_validation_tolerance = 2

        # bbox
        self.lost_tolerance = lost_tolerance
        self.last_target_bbox: tuple[int, int, int, int] | None = None
        self.last_target_frame_index: int | None = None
        self.target_center_velocity = np.zeros(2, dtype=np.float32)

        # target classifier
        self.classifier_collect_interval = 1
        self.classifier_max_overlap_iou = 0.2
        self.min_negatives = 30
        self.min_positives = 20

    def remember_target_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_index: int,
    ) -> None:
        """Update the recent target motion used for short-gap prediction."""
        current_bbox = np.asarray(bbox, dtype=np.float32)

        if (
            self.last_target_bbox is not None
            and self.last_target_frame_index is not None
        ):
            elapsed_frames = max(1, frame_index - self.last_target_frame_index)
            previous_bbox = np.asarray(self.last_target_bbox, dtype=np.float32)
            current_center = np.array(
                [
                    (current_bbox[0] + current_bbox[2]) / 2,
                    (current_bbox[1] + current_bbox[3]) / 2,
                ],
                dtype=np.float32,
            )
            previous_center = np.array(
                [
                    (previous_bbox[0] + previous_bbox[2]) / 2,
                    (previous_bbox[1] + previous_bbox[3]) / 2,
                ],
                dtype=np.float32,
            )
            measured_velocity = (current_center - previous_center) / elapsed_frames

            # Smooth detector jitter while retaining the athlete's fast motion.
            # New velocity = 65% * current measured velocity + 35% * previously stored velocity
            self.target_center_velocity = (
                0.65 * measured_velocity + 0.35 * self.target_center_velocity
            )

        self.last_target_bbox = bbox
        self.last_target_frame_index = frame_index

    def predict_target_bbox(
        self,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        """Predict a target box during a short detector/ID gap."""
        if self.last_target_bbox is None or self.last_target_frame_index is None:
            return None

        elapsed_frames = frame_index - self.last_target_frame_index
        if elapsed_frames <= 0 or elapsed_frames > self.lost_tolerance:
            return None

        x1, y1, x2, y2 = self.last_target_bbox
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return None

        previous_center = np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2],
            dtype=np.float32,
        )
        velocity_decay = 0.9
        decayed_displacement = (
            self.target_center_velocity
            * (1 - velocity_decay**elapsed_frames)
            / (1 - velocity_decay)
        )
        predicted_center = previous_center + decayed_displacement

        # Move only the center. Keep the last real detection's box dimensions,
        # shifting the whole box back inside the frame when it reaches an edge.
        predicted_x1 = int(np.rint(predicted_center[0] - box_width / 2))
        predicted_y1 = int(np.rint(predicted_center[1] - box_height / 2))
        predicted_x1 = int(np.clip(predicted_x1, 0, frame_width - box_width))
        predicted_y1 = int(np.clip(predicted_y1, 0, frame_height - box_height))
        predicted_x2 = predicted_x1 + box_width
        predicted_y2 = predicted_y1 + box_height

        if predicted_x2 <= predicted_x1 or predicted_y2 <= predicted_y1:
            return None

        return predicted_x1, predicted_y1, predicted_x2, predicted_y2

    def get_tracked_persons(self, result) -> list[TrackedPerson]:
        """Get person crops, bounding boxes, and IDs from one frame."""
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        frame = result.orig_img
        height, width = frame.shape[:2]
        xyxy = boxes.xyxy.detach().cpu().numpy()
        track_ids = boxes.id.detach().cpu().numpy().astype(int)

        tracked_persons = []
        for box, track_id in zip(xyxy, track_ids):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width))
            y2 = int(np.clip(y2, 0, height))

            if x2 <= x1 or y2 <= y1:
                continue

            tracked_persons.append(
                TrackedPerson(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    crop=frame[y1:y2, x1:x2].copy(),
                )
            )

        return tracked_persons

    def calculate_iou(
        self,
        first_bbox: tuple[int, int, int, int],
        second_bbox: tuple[int, int, int, int],
    ) -> float:
        """Calculate intersection over union for two xyxy boxes."""
        first_x1, first_y1, first_x2, first_y2 = first_bbox
        second_x1, second_y1, second_x2, second_y2 = second_bbox

        intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
        intersection_height = max(
            0, min(first_y2, second_y2) - max(first_y1, second_y1)
        )
        intersection_area = intersection_width * intersection_height

        first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
        second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
        union_area = first_area + second_area - intersection_area
        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    def update_feature_memory(
        self,
        other_persons: list[TrackedPerson],
        target: TrackedPerson,
        target_features: torch.Tensor,
    ) -> None:
        is_target_clean = True
        safe_negative_crops = []

        for person in other_persons:
            iou = self.calculate_iou(target.bbox, person.bbox)
            if iou <= self.classifier_max_overlap_iou:
                safe_negative_crops.append(person.crop)
            else:
                is_target_clean = False

        if safe_negative_crops:
            negative_features = self.person_feature_extractor.extract_features(
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

    def search_target_with_reid(
        self,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        if not tracked_persons:
            return None

        crops = [person.crop for person in tracked_persons]
        features = self.person_feature_extractor.extract_features(crops)

        if self.target_classifier.is_trained:
            best_index, best_score, _ = self.target_classifier.find_target(
                features.detach()
            )
        else:
            best_index, best_score, _ = self.target_gallery_matcher.find_target(
                features.detach(), self.positive_feature_memory
            )

        # if not self.target_classifier.is_trained or best_index is None:
        #     best_index, best_score, _ = self.target_gallery_matcher.find_target(
        #         features, self.positive_feature_memory
        #     )

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

    def keep_tracking_target(
        self,
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
                f"Frame {frame_index}: target ID missing, " f"switching to ReID search"
            )
            self.state = "SEARCHING"
            self.target_track_id = None
            self.reid_validation_failures = 0
            return self.search_target_with_reid(
                tracked_persons=tracked_persons,
                frame_index=frame_index,
            )

        # validate the target's appearance every N frames
        if frame_index % self.reid_validation_interval != 0:
            return target
        else:  # validate
            # calculate reid score
            target_features = self.person_feature_extractor.extract_features(
                target.crop
            )
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
                return self.search_target_with_reid(
                    tracked_persons=tracked_persons,
                    frame_index=frame_index,
                )

    def update_target_state(
        self,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """
        SEARCHING locks the best qualified global ReID candidate.
        TRACKING follows the locked ID and periodically validates appearance.
        """
        if self.state == "SEARCHING":
            return self.search_target_with_reid(tracked_persons, frame_index)

        if self.state == "TRACKING":
            return self.keep_tracking_target(tracked_persons, frame_index)

        raise RuntimeError(f"Unknown tracking state: {self.state}")

    def track_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output_path = output_path.with_name(
            f"{output_path.stem}.mp4v-temp.mp4"
        )

        # Get the FPS of the input video.
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0:
            fps = 30.0

        # Reset the tracking state.
        self.state = "SEARCHING"
        self.target_track_id = None
        self.missing_frames = 0
        self.last_target_bbox = None
        self.last_target_frame_index = None
        self.target_center_velocity.fill(0)
        self.reid_validation_failures = 0
        self.target_classifier.reset()

        # track
        results = self.tracker.track(input_path=input_path)
        writer = None

        progress_bar = tqdm(
            results,
            total=total_frames,
            desc="Tracking",
            unit="frame",
        )

        try:
            for frame_index, result in enumerate(progress_bar):
                # Track all people in the current frame.
                tracked_persons = self.get_tracked_persons(result)

                target = self.update_target_state(
                    tracked_persons,
                    frame_index,
                )

                # Remember real detections and bridge short detector/ID gaps
                # with a motion-predicted box.
                predicted_bbox = None
                if target is not None:
                    self.remember_target_bbox(target.bbox, frame_index)
                else:
                    height, width = result.orig_img.shape[:2]
                    predicted_bbox = self.predict_target_bbox(
                        frame_index,
                        frame_width=width,
                        frame_height=height,
                    )

                # Visualize the tracking result.
                frame = result.orig_img.copy()
                if target is not None:
                    x1, y1, x2, y2 = target.bbox
                    cv2.rectangle(
                        img=frame,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=(0, 255, 0),
                        thickness=3,
                    )
                    cv2.putText(
                        img=frame,
                        text=f"Target ID: {target.track_id}",
                        org=(x1, max(y1 - 10, 25)),  # origin of the text
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                elif predicted_bbox is not None:
                    x1, y1, x2, y2 = predicted_bbox
                    cv2.rectangle(
                        img=frame,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=(0, 165, 255),
                        thickness=3,
                    )
                    cv2.putText(
                        img=frame,
                        text="Target predicted",
                        org=(x1, max(y1 - 10, 25)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 165, 255),
                        thickness=2,
                    )

                cv2.putText(
                    img=frame,
                    text=self.state,
                    org=(20, 35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 255, 255),
                    thickness=2,
                )

                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(temporary_output_path),
                        fourcc,
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Unable to create temporary video: "
                            f"{temporary_output_path}"
                        )

                writer.write(frame)
        finally:
            progress_bar.close()
            if writer is not None:
                writer.release()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temporary_output_path),
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output_path),
            ],
            check=True,
        )
        temporary_output_path.unlink()

        print(f"Tracking result saved to: {output_path}")
