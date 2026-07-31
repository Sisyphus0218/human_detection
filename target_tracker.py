import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from target_reid import TargetReID


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    crop: np.ndarray


class TargetTracker:
    def __init__(
        self,
        yolo_path: str | Path,
        tracker_config: str,
        reid: TargetReID,
        device: str,
        confidence: float = 0.5,
        lost_tolerance: int = 5,
        image_size: int = 1280,
    ) -> None:
        self.model = YOLO(yolo_path)
        self.tracker_config = tracker_config
        self.reid = reid
        self.device = device

        self.confidence = confidence
        self.imgsz = image_size

        # ReID state-machine parameters.
        self.state = "SEARCHING"
        self.target_track_id: int | None = None
        # Keep this limit only for short-gap bounding-box prediction.
        self.lost_tolerance = lost_tolerance

        # SEARCHING
        self.reid_search_threshold = self.reid.threshold  # best score >= threshold
        self.reid_margin_threshold = 0.0  # best score - second score >= threshold

        self.pending_track_id: int | None = None
        self.pending_hits = 0
        self.search_confirm_frames = 3

        # TRACKING
        self.reid_keep_threshold = 0.50
        self.reid_reject_threshold = 0.40

        self.reid_validation_interval = 5
        self.reid_validation_failures = 0
        self.reid_validation_tolerance = 2

        self.last_target_bbox: tuple[int, int, int, int] | None = None
        self.last_target_frame_index: int | None = None
        self.target_center_velocity = np.zeros(2, dtype=np.float32)

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

    def get_person_by_id(
        self,
        tracked_persons: list[TrackedPerson],
    ) -> TrackedPerson | None:
        """Return the locked track without changing tracking state."""
        if self.target_track_id is None:
            return None

        for person in tracked_persons:
            if person.track_id == self.target_track_id:
                return person
        return None

    def get_best_reid_candidate(
        self,
        tracked_persons: list[TrackedPerson],
    ) -> tuple[TrackedPerson | None, float, float]:
        """Return the best candidate, its score, and its lead over second."""
        if not tracked_persons:
            return None, 0.0, 0.0

        crops = [person.crop for person in tracked_persons]
        scores = self.reid.calculate_scores(crops)
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_score = float(scores[best_index])

        if len(order) > 1:
            second_score = float(scores[int(order[1])])
            score_margin = best_score - second_score
        else:
            score_margin = float("inf")

        return tracked_persons[best_index], best_score, score_margin

    def clear_pending_candidate(self) -> None:
        """Discard an unconfirmed ReID candidate."""
        self.pending_track_id = None
        self.pending_hits = 0

    def search_target_with_reid(
        self,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """Search globally with a strict three-frame confirmation."""
        # get the target candidate
        candidate, score, margin = self.get_best_reid_candidate(tracked_persons)

        if candidate is None:
            self.clear_pending_candidate()
            return None
        else:
            qualified = (
                score >= self.reid_search_threshold  # best score >= threshold
                and margin
                >= self.reid_margin_threshold  # best score - second score >= threshold
            )

            if not qualified:
                self.clear_pending_candidate()
                return None

            # if the candidate is the same as the previous frame, increment the hit count
            if candidate.track_id == self.pending_track_id:
                self.pending_hits += 1
            else:
                self.pending_track_id = candidate.track_id
                self.pending_hits = 1

            if self.pending_hits < self.search_confirm_frames:
                return None
            else:  # the candidate has been confirmed for the required number of frames
                self.state = "TRACKING"
                self.target_track_id = candidate.track_id
                self.reid_validation_failures = 0
                self.clear_pending_candidate()

                tqdm.write(
                    f"Frame {frame_index}: target confirmed, "
                    f"track_id={candidate.track_id}, "
                    f"ReID score={score:.4f}"
                )
                return candidate

    def update_tracking_target(
        self,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """Follow the locked ID and periodically validate its appearance."""
        target = self.get_person_by_id(tracked_persons)

        # Once the locked ID disappears, immediately search all current tracks
        # with ReID instead of waiting for a lost-track tolerance window.
        if target is None:
            tqdm.write(
                f"Frame {frame_index}: target ID missing, " f"switching to ReID search"
            )
            self.state = "SEARCHING"
            self.target_track_id = None
            self.reid_validation_failures = 0
            self.clear_pending_candidate()
            return self.search_target_with_reid(
                tracked_persons=tracked_persons,
                frame_index=frame_index,
            )

        # target is present, validate its appearance every N frames
        if frame_index % self.reid_validation_interval != 0:
            return target

        target_score = float(self.reid.calculate_scores([target.crop])[0])

        if target_score >= self.reid_keep_threshold:
            self.reid_validation_failures = 0
            return target
        elif target_score < self.reid_reject_threshold:
            self.reid_validation_failures = self.reid_validation_tolerance
        else:
            self.reid_validation_failures += 1

        if self.reid_validation_failures < self.reid_validation_tolerance:
            return target

        tqdm.write(
            f"Frame {frame_index}: locked ID failed ReID validation, "
            f"track_id={target.track_id}, "
            f"ReID score={target_score:.4f}"
        )
        self.state = "SEARCHING"
        self.target_track_id = None
        self.reid_validation_failures = 0
        self.clear_pending_candidate()
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
        SEARCHING confirms a global ReID candidate for three frames.
        TRACKING follows the locked ID and periodically validates appearance.
        """
        if self.state == "SEARCHING":
            return self.search_target_with_reid(tracked_persons, frame_index)

        if self.state == "TRACKING":
            return self.update_tracking_target(tracked_persons, frame_index)

        raise RuntimeError(f"Unknown tracking state: {self.state}")

    def track_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Track the registered target and save an annotated result video."""
        if self.reid.gallery_size == 0:
            raise RuntimeError("Please register the target before starting tracking")

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

        # Initialize the tracker.
        track_kwargs = {
            "source": input_path,
            "classes": [0],
            "tracker": self.tracker_config,
            "device": self.device,
            "conf": self.confidence,
            "imgsz": self.imgsz,
            "stream": True,
            "persist": True,
            "verbose": False,
        }

        results = self.model.track(**track_kwargs)
        writer = None

        # Reset the tracking state.
        self.state = "SEARCHING"
        self.target_track_id = None
        self.missing_frames = 0
        self.last_target_bbox = None
        self.last_target_frame_index = None
        self.target_center_velocity.fill(0)
        self.clear_pending_candidate()
        self.reid_validation_failures = 0

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
