import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from logistic_target_classifier import LogisticTargetClassifier
from person_feature_extractor import PersonFeatureExtractor
from feature_memory import FeatureMemory
from person_tracker import PersonTracker
from target_gallery_matcher import TargetGalleryMatcher


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    crop: np.ndarray


@dataclass(frozen=True)
class TrackingPipelineConfig:
    reid_keep_threshold: float
    reid_reject_threshold: float
    reid_validation_interval: int
    reid_validation_tolerance: int
    missing_frame_tolerance: int
    lost_tolerance: int
    velocity_smoothing: float
    velocity_decay: float
    classifier_collect_interval: int
    classifier_max_overlap_iou: float
    min_negatives: int
    min_positives: int


def build_dense_bbox_trajectory(
    frame_bboxes: list[tuple[int, int, int, int] | None],
) -> torch.Tensor:
    """Fill missing per-frame boxes without changing the tracking state."""
    frame_count = len(frame_bboxes)
    if frame_count == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    valid_indices = np.asarray(
        [index for index, bbox in enumerate(frame_bboxes) if bbox is not None],
        dtype=np.int64,
    )
    if len(valid_indices) == 0:
        return torch.full((frame_count, 4), float("nan"), dtype=torch.float32)

    valid_bboxes = np.asarray(
        [frame_bboxes[index] for index in valid_indices],
        dtype=np.float32,
    )
    all_indices = np.arange(frame_count, dtype=np.float32)
    dense_bboxes = np.empty((frame_count, 4), dtype=np.float32)
    for coordinate_index in range(4):
        dense_bboxes[:, coordinate_index] = np.interp(
            all_indices,
            valid_indices,
            valid_bboxes[:, coordinate_index],
        )

    return torch.from_numpy(dense_bboxes)


class TargetTrackingPipeline:
    def __init__(
        self,
        tracker: PersonTracker,
        person_feature_extractor: PersonFeatureExtractor,
        positive_feature_memory: FeatureMemory,
        negative_feature_memory: FeatureMemory,
        target_gallery_matcher: TargetGalleryMatcher,
        target_classifier: LogisticTargetClassifier,
        config: TrackingPipelineConfig,
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

        self.reid_keep_threshold = config.reid_keep_threshold
        self.reid_reject_threshold = config.reid_reject_threshold

        self.reid_validation_interval = config.reid_validation_interval
        self.reid_validation_tolerance = config.reid_validation_tolerance
        self.reid_validation_failures = 0

        self.missing_frame_tolerance = config.missing_frame_tolerance
        self.missing_frames = 0

        # bbox
        self.lost_tolerance = config.lost_tolerance
        self.velocity_smoothing = config.velocity_smoothing
        self.velocity_decay = config.velocity_decay
        self.last_target_bbox: tuple[int, int, int, int] | None = None
        self.last_target_frame_index: int | None = None
        self.target_center_velocity = np.zeros(2, dtype=np.float32)

        # target classifier
        self.classifier_collect_interval = config.classifier_collect_interval
        self.classifier_max_overlap_iou = config.classifier_max_overlap_iou
        self.min_negatives = config.min_negatives
        self.min_positives = config.min_positives

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
            previous_velocity_weight = 1 - self.velocity_smoothing
            self.target_center_velocity = (
                self.velocity_smoothing * measured_velocity
                + previous_velocity_weight * self.target_center_velocity
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
        decayed_displacement = (
            self.target_center_velocity
            * (1 - self.velocity_decay**elapsed_frames)
            / (1 - self.velocity_decay)
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
                f"Frame {frame_index}: target ID missing, " f"switch to RECOVERING"
            )
            self.state = "RECOVERING"
            self.reid_validation_failures = 0
            return self.recover_target(
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

    def recover_target(
        self,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        if self.missing_frames >= self.missing_frame_tolerance:
            self.state = "SEARCHING"
            self.target_track_id = None
            self.missing_frames = 0
            return self.search_target_with_reid(
                tracked_persons=tracked_persons,
                frame_index=frame_index,
            )

        if len(tracked_persons) == 0:
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

        crops = [person.crop for person in tracked_persons]
        features = self.person_feature_extractor.extract_features(crops)
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

        if self.state == "RECOVERING":
            return self.recover_target(tracked_persons, frame_index)

        raise RuntimeError(f"Unknown tracking state: {self.state}")

    def render_tracking_frame(
        self,
        result,
        target: TrackedPerson | None,
        predicted_bbox: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        """Render the target or its short-gap prediction."""
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
                org=(x1, max(y1 - 10, 25)),
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
        return frame

    def render_debug_frame(
        self,
        result,
        frame_index: int,
        target: TrackedPerson | None,
    ) -> np.ndarray:
        """Render all raw tracker outputs and the current target state."""
        frame = result.plot(conf=True, labels=True)
        selected_id = target.track_id if target is not None else None

        frame_height, frame_width = frame.shape[:2]
        panel_top = max(0, frame_height - 44)
        panel = frame.copy()
        cv2.rectangle(
            img=panel,
            pt1=(0, panel_top),
            pt2=(frame_width, frame_height),
            color=(0, 0, 0),
            thickness=-1,
        )
        cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

        cv2.putText(
            img=frame,
            text=(
                f"Frame {frame_index} | State: {self.state} | "
                f"Selected ID: {selected_id}"
            ),
            org=(12, panel_top + 28),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 255, 255),
            thickness=2,
        )

        if target is not None:
            x1, y1, x2, y2 = target.bbox
            cv2.rectangle(
                img=frame,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(255, 0, 255),
                thickness=4,
            )
            cv2.putText(
                img=frame,
                text=f"ReID selected ID: {target.track_id}",
                org=(x1, max(y1 - 12, 145)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 0, 255),
                thickness=2,
            )

        return frame

    @staticmethod
    def convert_video_to_h264(
        temporary_path: Path,
        output_path: Path,
    ) -> None:
        """Convert a temporary MP4V video to an H.264 MP4 video."""
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temporary_path),
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
        temporary_path.unlink()

    def track_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        bbox_output_path: str | Path,
        debug_output_path: str | Path,
        bbox_enabled: bool = True,
        debug_enabled: bool = True,
    ) -> None:
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output_path = output_path.with_name(
            f"{output_path.stem}.mp4v-temp.mp4"
        )

        debug_temporary_output_path = None
        if debug_enabled:
            debug_output_path = Path(debug_output_path)
            debug_output_path.parent.mkdir(parents=True, exist_ok=True)
            debug_temporary_output_path = debug_output_path.with_name(
                f"{debug_output_path.stem}.mp4v-temp.mp4"
            )

        if bbox_enabled:
            bbox_output_path = Path(bbox_output_path)
            bbox_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the FPS of the input video.
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        debug_writer = None
        frame_bboxes: list[tuple[int, int, int, int] | None] = []
        bbox_sources: list[int] = []
        track_ids: list[int] = []

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

                if bbox_enabled:
                    if target is not None:
                        frame_bboxes.append(target.bbox)
                        bbox_sources.append(2)  # observed
                        track_ids.append(target.track_id)
                    elif predicted_bbox is not None:
                        frame_bboxes.append(predicted_bbox)
                        bbox_sources.append(1)  # predicted
                        track_ids.append(-1)
                    else:
                        frame_bboxes.append(None)
                        bbox_sources.append(0)  # interpolated when exported
                        track_ids.append(-1)

                frame = self.render_tracking_frame(
                    result=result,
                    target=target,
                    predicted_bbox=predicted_bbox,
                )

                # Save a second view containing every raw YOLO+tracker result.
                # ReID only selects a target and does not modify ``result``, so
                # this is the exact multi-person tracking input seen by ReID.
                debug_frame = None
                if debug_enabled:
                    debug_frame = self.render_debug_frame(
                        result=result,
                        frame_index=frame_index,
                        target=target,
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

                if debug_frame is not None:
                    if debug_writer is None:
                        height, width = debug_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        debug_writer = cv2.VideoWriter(
                            str(debug_temporary_output_path),
                            fourcc,
                            fps,
                            (width, height),
                        )
                        if not debug_writer.isOpened():
                            raise RuntimeError(
                                f"Unable to create temporary debug video: "
                                f"{debug_temporary_output_path}"
                            )
                    debug_writer.write(debug_frame)
        finally:
            progress_bar.close()
            if writer is not None:
                writer.release()
            if debug_writer is not None:
                debug_writer.release()

        self.convert_video_to_h264(
            temporary_path=temporary_output_path,
            output_path=output_path,
        )

        if debug_enabled:
            self.convert_video_to_h264(
                temporary_path=debug_temporary_output_path,
                output_path=debug_output_path,
            )

        if bbox_enabled:
            bbx_xyxy = build_dense_bbox_trajectory(frame_bboxes)
            bbox_source = torch.tensor(bbox_sources, dtype=torch.uint8)
            bbox_observed_mask = bbox_source == 2
            bbox_preinterpolation_mask = bbox_source != 0
            torch.save(
                {
                    "bbx_xyxy": bbx_xyxy,
                    "bbox_source": bbox_source,
                    "bbox_source_names": {
                        0: "interpolated",
                        1: "predicted",
                        2: "observed",
                    },
                    "bbox_observed_mask": bbox_observed_mask,
                    "bbox_preinterpolation_mask": bbox_preinterpolation_mask,
                    "track_id": torch.tensor(track_ids, dtype=torch.int64),
                    "bbox_format": "xyxy",
                    "coordinate_space": "original_video_pixels",
                    "video_path": str(input_path.resolve()),
                    "num_frames": len(frame_bboxes),
                    "video_width": video_width,
                    "video_height": video_height,
                    "fps": float(fps),
                },
                bbox_output_path,
            )

        print(f"Tracking result saved to: {output_path}")
        if debug_enabled:
            print(f"All-tracks debug video saved to: {debug_output_path}")
        if bbox_enabled:
            print(f"Bounding boxes saved to: {bbox_output_path}")
