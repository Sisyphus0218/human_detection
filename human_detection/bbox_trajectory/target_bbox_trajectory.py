from pathlib import Path

import numpy as np
import torch

from human_detection.frame_source import FrameSource
from human_detection.person_tracker import TrackedPerson

from .bbox_trajectory_entry import BBoxSource, BBoxTrajectoryEntry


class TargetBBoxTrajectory:
    """Track observed, predicted, and missing target bounding boxes."""

    _SOURCE_CODES = {
        BBoxSource.MISSING: 0,
        BBoxSource.PREDICTED: 1,
        BBoxSource.OBSERVED: 2,
    }

    def __init__(
        self,
        lost_tolerance: int,
        velocity_smoothing: float,
        velocity_decay: float,
    ) -> None:
        self.lost_tolerance = lost_tolerance
        self.velocity_smoothing = velocity_smoothing
        self.velocity_decay = velocity_decay

        self.entries: list[BBoxTrajectoryEntry] = []

        self.last_observed_bbox: tuple[int, int, int, int] | None = None
        self.last_observed_frame_index: int | None = None
        self.center_velocity = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.entries.clear()
        self.last_observed_bbox = None
        self.last_observed_frame_index = None
        self.center_velocity.fill(0)

    def update(
        self,
        frame_index: int,
        target: TrackedPerson | None,
        frame_width: int,
        frame_height: int,
    ) -> BBoxTrajectoryEntry:
        if target is not None:
            self.remember_bbox(target.bbox, frame_index)
            entry = BBoxTrajectoryEntry(
                frame_index=frame_index,
                bbox=target.bbox,
                source=BBoxSource.OBSERVED,
                track_id=target.track_id,
            )
        else:
            predicted_bbox = self.predict_bbox(
                frame_index=frame_index,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            entry = BBoxTrajectoryEntry(
                frame_index=frame_index,
                bbox=predicted_bbox,
                source=(
                    BBoxSource.PREDICTED
                    if predicted_bbox is not None
                    else BBoxSource.MISSING
                ),
                track_id=None,
            )

        self.entries.append(entry)
        return entry

    def remember_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_index: int,
    ) -> None:
        current_bbox = np.asarray(bbox, dtype=np.float32)

        if (
            self.last_observed_bbox is not None
            and self.last_observed_frame_index is not None
        ):
            elapsed_frames = max(1, frame_index - self.last_observed_frame_index)
            previous_bbox = np.asarray(self.last_observed_bbox, dtype=np.float32)
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
            self.center_velocity = (
                self.velocity_smoothing * measured_velocity
                + (1 - self.velocity_smoothing) * self.center_velocity
            )

        self.last_observed_bbox = bbox
        self.last_observed_frame_index = frame_index

    def predict_bbox(
        self,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        if self.last_observed_bbox is None or self.last_observed_frame_index is None:
            return None

        elapsed_frames = frame_index - self.last_observed_frame_index
        if elapsed_frames <= 0 or elapsed_frames > self.lost_tolerance:
            return None

        x1, y1, x2, y2 = self.last_observed_bbox
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return None

        previous_center = np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2],
            dtype=np.float32,
        )
        if self.velocity_decay == 1:
            decayed_displacement = self.center_velocity * elapsed_frames
        else:
            decayed_displacement = (
                self.center_velocity
                * (1 - self.velocity_decay**elapsed_frames)
                / (1 - self.velocity_decay)
            )
        predicted_center = previous_center + decayed_displacement

        predicted_x1 = int(np.rint(predicted_center[0] - box_width / 2))
        predicted_y1 = int(np.rint(predicted_center[1] - box_height / 2))
        predicted_x1 = int(np.clip(predicted_x1, 0, frame_width - box_width))
        predicted_y1 = int(np.clip(predicted_y1, 0, frame_height - box_height))
        predicted_x2 = predicted_x1 + box_width
        predicted_y2 = predicted_y1 + box_height

        if predicted_x2 <= predicted_x1 or predicted_y2 <= predicted_y1:
            return None

        return predicted_x1, predicted_y1, predicted_x2, predicted_y2

    def dense_bboxes(self) -> torch.Tensor:
        frame_count = len(self.entries)
        if frame_count == 0:
            return torch.empty((0, 4), dtype=torch.float32)

        valid_indices = np.asarray(
            [
                index
                for index, entry in enumerate(self.entries)
                if entry.bbox is not None
            ],
            dtype=np.int64,
        )
        if len(valid_indices) == 0:
            return torch.full((frame_count, 4), float("nan"), dtype=torch.float32)

        valid_bboxes = np.asarray(
            [self.entries[index].bbox for index in valid_indices],
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

    def save(self, output_path: str | Path, source: FrameSource) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bbox_source = torch.tensor(
            [self._SOURCE_CODES[entry.source] for entry in self.entries],
            dtype=torch.uint8,
        )
        torch.save(
            {
                "bbx_xyxy": self.dense_bboxes(),
                "bbox_source": bbox_source,
                "bbox_source_names": {
                    0: "interpolated",
                    1: "predicted",
                    2: "observed",
                },
                "bbox_observed_mask": bbox_source == 2,
                "bbox_preinterpolation_mask": bbox_source != 0,
                "track_id": torch.tensor(
                    [
                        entry.track_id if entry.track_id is not None else -1
                        for entry in self.entries
                    ],
                    dtype=torch.int64,
                ),
                "bbox_format": "xyxy",
                "coordinate_space": "source_frame_pixels",
                "source_type": type(source).__name__,
                "num_frames": len(self.entries),
                "frame_width": source.width,
                "frame_height": source.height,
                "fps": float(source.fps),
            },
            output_path,
        )
