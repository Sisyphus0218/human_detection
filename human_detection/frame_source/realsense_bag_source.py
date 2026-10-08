from pathlib import Path

import cv2
import numpy as np

from .frame_source import FrameSource
from .rgbd_frame import RGBDFrame


class RealSenseBagSource(FrameSource):
    """Replay aligned RGB-D frames from a librealsense recording."""

    def __init__(
        self,
        name: str,
        bag_path: str | Path,
        real_time: bool = False,
        timeout_ms: int = 5_000,
    ) -> None:
        super().__init__(name=name)
        self.bag_path = Path(bag_path).resolve()
        if not self.bag_path.is_file():
            raise FileNotFoundError(f"RealSense recording not found: {self.bag_path}")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

        self.real_time = real_time
        self.timeout_ms = timeout_ms

        self._rs = None
        self._pipeline = None
        self._playback = None
        self._align = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._intrinsics: np.ndarray | None = None
        self._frame_index = 0
        self._is_open = False

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
        # Recordings may contain dropped or unsynchronised streams, so a duration-based
        # estimate would be misleading.
        return None

    @property
    def intrinsics(self) -> np.ndarray | None:
        return self._intrinsics

    def open(self) -> None:
        if self._is_open:
            return

        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "RealSense Python bindings are not installed. "
                "Install the 'pyrealsense2' package in the active environment."
            ) from error

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(self.bag_path), repeat_playback=False)

        try:
            profile = pipeline.start(config)
            playback = profile.get_device().as_playback()
            playback.set_real_time(self.real_time)

            color_profile = profile.get_stream(rs.stream.color)
            color_profile = color_profile.as_video_stream_profile()
            color_intrinsics = color_profile.get_intrinsics()

            self._rs = rs
            self._pipeline = pipeline
            self._playback = playback
            self._align = rs.align(rs.stream.color)
            self._width = color_profile.width()
            self._height = color_profile.height()
            self._fps = float(color_profile.fps())
            self._intrinsics = np.asarray(
                [
                    [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
                    [0.0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self._frame_index = 0
            self._is_open = True
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise

    def read(self) -> RGBDFrame | None:
        if not self._is_open or self._pipeline is None:
            raise RuntimeError("RealSense recording is not open")

        while True:
            try:
                frames = self._pipeline.wait_for_frames(self.timeout_ms)
            except RuntimeError as error:
                if self._playback_has_stopped():
                    return None
                raise RuntimeError(
                    f"Unable to read RealSense recording: {self.bag_path}"
                ) from error

            aligned_frames = self._align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                if self._playback_has_stopped():
                    return None
                continue

            color_bgr = self._color_to_bgr(color_frame)
            depth_mm = self._depth_to_mm(depth_frame)
            timestamp_ms = float(color_frame.get_timestamp())

            frame = RGBDFrame(
                color_bgr=color_bgr,
                depth_mm=depth_mm,
                frame_index=self._frame_index,
                timestamp_ms=timestamp_ms,
            )
            self._frame_index += 1
            return frame

    def _playback_has_stopped(self) -> bool:
        return (
            self._playback is not None
            and self._rs is not None
            and self._playback.current_status() == self._rs.playback_status.stopped
        )

    def _color_to_bgr(self, color_frame) -> np.ndarray:
        color = np.asanyarray(color_frame.get_data())
        color_format = color_frame.profile.format()

        if color_format == self._rs.format.bgr8:
            return color.copy()
        if color_format == self._rs.format.rgb8:
            return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        if color_format == self._rs.format.bgra8:
            return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        if color_format == self._rs.format.rgba8:
            return cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

        raise RuntimeError(f"Unsupported RealSense color format: {color_format}")

    @staticmethod
    def _depth_to_mm(depth_frame) -> np.ndarray:
        depth = np.asanyarray(depth_frame.get_data())
        millimeters_per_unit = float(depth_frame.get_units()) * 1_000.0
        if np.isclose(millimeters_per_unit, 1.0):
            return depth.copy()
        return depth.astype(np.float32) * millimeters_per_unit

    def close(self) -> None:
        if self._pipeline is not None and self._is_open:
            try:
                self._pipeline.stop()
            except Exception:
                pass

        self._pipeline = None
        self._playback = None
        self._align = None
        self._is_open = False
