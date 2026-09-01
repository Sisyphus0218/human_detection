from pathlib import Path

import cv2

from .frame_source import FrameSource
from .rgbd_frame import RGBDFrame


class VideoFileSource(FrameSource):
    """Read color frames from a video file."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path).resolve()
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video file: {self.video_path}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise RuntimeError(f"Unable to read video metadata: {self.video_path}")
        finally:
            capture.release()

        self._capture: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._is_open = False

        self._width = width
        self._height = height
        self._fps = float(fps)
        self._frame_count = frame_count if frame_count > 0 else None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int | None:
        return self._frame_count

    def open(self) -> None:
        if self._is_open:
            return

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video file: {self.video_path}")

        self._capture = capture
        self._frame_index = 0
        self._is_open = True

    def read(self) -> RGBDFrame | None:
        if not self._is_open or self._capture is None:
            raise RuntimeError("Video source is not open")

        success, color_bgr = self._capture.read()
        if not success:
            return None

        timestamp_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)

        frame = RGBDFrame(
            color_bgr=color_bgr,
            depth_mm=None,
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
        )

        self._frame_index += 1

        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()

        self._capture = None
        self._is_open = False
