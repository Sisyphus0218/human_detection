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
        self.lost_tolerance = lost_tolerance
        self.imgsz = image_size

        self.target_track_id: int | None = None
        self.state = "SEARCHING"
        self.missing_frames = 0
        self.last_target_bbox: tuple[int, int, int, int] | None = None
        self.last_target_frame_index: int | None = None
        self.target_bbox_velocity = np.zeros(4, dtype=np.float32)

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
            measured_velocity = (current_bbox - previous_bbox) / elapsed_frames

            # Smooth detector jitter while retaining the athlete's fast motion.
            # New velocity = 65% * current measured velocity + 35% * previously stored velocity
            self.target_bbox_velocity = (
                0.65 * measured_velocity + 0.35 * self.target_bbox_velocity
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

        previous_bbox = np.asarray(self.last_target_bbox, dtype=np.float32)
        velocity_decay = 0.9
        decayed_displacement = (
            self.target_bbox_velocity
            * (1 - velocity_decay**elapsed_frames)
            / (1 - velocity_decay)
        )
        predicted_bbox = previous_bbox + decayed_displacement

        x1, y1, x2, y2 = np.rint(predicted_bbox).astype(int)
        x1 = int(np.clip(x1, 0, frame_width - 1))
        y1 = int(np.clip(y1, 0, frame_height - 1))
        x2 = int(np.clip(x2, 0, frame_width))
        y2 = int(np.clip(y2, 0, frame_height))
        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

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

    def find_target_with_reid(
        self,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """Search for the target with ReID and lock its current track ID."""
        self.state = "SEARCHING"

        if not tracked_persons:
            return None

        crops = [person.crop for person in tracked_persons]
        target_index, target_score, _ = self.reid.find_target(crops)
        if target_index is None:
            return None

        target = tracked_persons[target_index]
        self.target_track_id = target.track_id
        self.missing_frames = 0
        self.state = "TRACKING"

        tqdm.write(
            f"Frame {frame_index}: target found, "
            f"track_id={target.track_id}, "
            f"ReID score={target_score:.4f}"
        )

        return target

    def find_target_by_id(
        self,
        tracked_persons: list[TrackedPerson],
    ) -> TrackedPerson | None:
        """Find the person whose ID matches the currently locked target ID."""
        for person in tracked_persons:
            if person.track_id == self.target_track_id:
                self.state = "TRACKING"
                self.missing_frames = 0
                return person

        self.state = "MISSING"
        self.missing_frames += 1
        return None

    def update_target_state(
        self,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """
        START
        ↓
        SEARCHING
        ├─ Target not found by ReID ─────────→ SEARCHING
        └─ Target found by ReID ─────────────→ TRACKING

        TRACKING
        ├─ Target ID is present ─────────────→ TRACKING
        └─ Target ID is missing ─────────────→ MISSING

        MISSING
        ├─ Previous ID reappears while missing_frames < lost_tolerance ──→ TRACKING
        ├─ Target ID remains missing while missing_frames < lost_tolerance ──→ MISSING
        └─ missing_frames >= lost_tolerance
                ↓
            Unlock the previous ID
                ↓
            SEARCHING
        """

        # SEARCHING: No target ID is currently locked.
        # Use ReID to find the target among the current tracked people.
        if self.state == "SEARCHING":
            target = self.find_target_with_reid(tracked_persons, frame_index)
            return target

        # TRACKING: The target ID is currently locked.
        # Keep following the currently locked target ID.
        elif self.state == "TRACKING":
            target = self.find_target_by_id(tracked_persons)
            return target

        # MISSING: The target ID is currently locked, but the target is not detected.
        elif self.state == "MISSING":
            if self.missing_frames < self.lost_tolerance:
                target = self.find_target_by_id(tracked_persons)
                return target

            # The old ID has been missing for too long. Unlock the old ID.
            # Use ReID to search for the target under a possible new track ID.
            self.target_track_id = None
            target = self.find_target_with_reid(tracked_persons, frame_index)
            return target

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
        self.target_track_id = None
        self.state = "SEARCHING"
        self.missing_frames = 0
        self.last_target_bbox = None
        self.last_target_frame_index = None
        self.target_bbox_velocity.fill(0)

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
