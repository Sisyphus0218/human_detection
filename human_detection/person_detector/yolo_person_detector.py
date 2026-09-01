from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .person_detector import PersonDetector


class YOLOPersonDetector(PersonDetector):
    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect_target(
        self,
        target_image_dir: str | Path,
        target_crop_dir: str | Path,
    ) -> list[np.ndarray]:
        """
        Crop the target from each image in the target_image_dir.
        Save the cropped images to target_crop_dir.

        Args:
            target_image_dir: Directory containing the images to process.
            target_crop_dir: Directory to save the cropped images.

        Returns:
            crops: A list of cropped BGR images.
        """

        # Check the validity of the directory.
        target_image_dir = Path(target_image_dir)
        if not target_image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {target_image_dir}")
        if not target_image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {target_image_dir}")

        target_crop_dir = Path(target_crop_dir)
        target_crop_dir.mkdir(parents=True, exist_ok=True)

        # Use YOLO to detect persons in each image.
        predict_kwargs = {
            "source": target_image_dir,
            "classes": [0],  # Class 0 in COCO is person.
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
        }
        results = self.model.predict(**predict_kwargs)

        # Process the results and save the cropped images to the target_crop_dir.
        crops: list[np.ndarray] = []
        processed_count = 0

        for result in results:  # results for all images, result for a single image
            processed_count += 1

            if result.boxes is None or len(result.boxes) == 0:
                print(f"Skipped {Path(result.path).name}: No person detected.")
                continue

            # Find the person with the highest confidence score.
            target_index = int(result.boxes.conf.argmax())
            box = result.boxes.xyxy[target_index].cpu().numpy()

            # Crop the detected person and save it to the target_crop_dir
            image = result.orig_img  # bgr
            height, width = image.shape[:2]

            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width))
            y2 = int(np.clip(y2, 0, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2].copy()  # bgr

            crop_path = target_crop_dir / Path(result.path).name
            if not cv2.imwrite(crop_path, crop):
                raise RuntimeError(f"Failed to save cropped image: {crop_path}")

            crops.append(crop)

        print(f"Processed images: {processed_count}")
        print(f"Detected and saved: {len(crops)}")
        print(f"Skipped images: {processed_count - len(crops)}")
        print(f"Cropped images saved to: {target_crop_dir.resolve()}")

        return crops
