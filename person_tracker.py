from pathlib import Path

from ultralytics import YOLO


class PersonTracker:
    def __init__(
        self,
        model_path: str | Path,
        tracker_config: dict,
        confidence: float = 0.25,
        image_size: int = 1280,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def track(
        self,
        input_path: str | Path,
    ):
        track_kwargs = {
            "source": input_path,
            "classes": [0],
            "tracker": self.tracker_config,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "device": self.device,
            "stream": True,
            "persist": True,
            "verbose": False,
        }

        results = self.model.track(**track_kwargs)
        return results
