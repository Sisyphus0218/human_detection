from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .person_tracker import PersonTracker
from .tracked_person import TrackedPerson


class YOLOPersonTracker(PersonTracker):
    def __init__(
        self,
        model_path: str | Path,
        tracking_algorithm: str | Path,
        confidence: float = 0.25,
        image_size: int = 1280,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.tracking_algorithm = tracking_algorithm
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def track(
        self,
        frame: np.ndarray,
    ) -> list[TrackedPerson]:
        track_kwargs = {
            "source": frame,
            "classes": [0],
            "tracker": self.tracking_algorithm,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "device": self.device,
            "stream": False,
            "persist": True,
            "verbose": False,
        }

        result = self.model.track(**track_kwargs)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        xyxy = boxes.xyxy.detach().cpu().numpy()
        track_ids = boxes.id.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()

        tracked_persons = []
        for box, track_id, confidence in zip(xyxy, track_ids, confidences):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1 = int(np.clip(x1, 0, frame_width - 1))
            y1 = int(np.clip(y1, 0, frame_height - 1))
            x2 = int(np.clip(x2, 0, frame_width))
            y2 = int(np.clip(y2, 0, frame_height))

            if x2 <= x1 or y2 <= y1:
                continue

            tracked_persons.append(
                TrackedPerson(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                )
            )

        return tracked_persons
