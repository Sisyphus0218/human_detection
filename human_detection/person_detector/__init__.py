"""Person detector interface and implementations."""

from .person_detector import PersonDetector
from .yolo_person_detector import YOLOPersonDetector

__all__ = ["PersonDetector", "YOLOPersonDetector"]
