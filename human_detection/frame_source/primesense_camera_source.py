from pathlib import Path

import cv2
import numpy as np

from .frame_source import FrameSource
from .rgbd_frame import RGBDFrame


class PrimeSenseCameraSource(FrameSource):
    """Read registered RGB-D frames from a PrimeSense camera via OpenNI2."""

    def __init__(
        self,
        openni2_redist_path: str | Path | None = None,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
    ) -> None:
        if openni2_redist_path is not None:
            self.openni2_redist_path = Path(openni2_redist_path)
        else:
            self.openni2_redist_path = None

        self._width = width
        self._height = height
        self._fps = fps

        self._openni2 = None
        self._device = None
        self._color_stream = None
        self._depth_stream = None
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
        return float(self._fps)

    @property
    def frame_count(self) -> int | None:
        return None

    def open(self) -> None:
        if self._is_open:
            return

        try:
            from primesense import _openni2 as c_api
            from primesense import openni2
        except ImportError as error:
            raise RuntimeError(
                "PrimeSense Python bindings are not installed. "
                "Install the 'primesense' package in the active environment."
            ) from error

        self._openni2 = openni2

        try:
            if self.openni2_redist_path is None:
                openni2.initialize()
            else:
                redist_path = self.openni2_redist_path.resolve()
                if not redist_path.is_dir():
                    raise FileNotFoundError(
                        f"OpenNI2 Redist directory not found: {redist_path}"
                    )
                openni2.initialize(str(redist_path))

            self._device = openni2.Device.open_any()
            self._color_stream = self._device.create_color_stream()
            self._depth_stream = self._device.create_depth_stream()

            self._color_stream.set_video_mode(
                c_api.OniVideoMode(
                    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                    resolutionX=self.width,
                    resolutionY=self.height,
                    fps=int(self.fps),
                )
            )
            self._depth_stream.set_video_mode(
                c_api.OniVideoMode(
                    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                    resolutionX=self.width,
                    resolutionY=self.height,
                    fps=int(self.fps),
                )
            )

            self._device.set_image_registration_mode(
                c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR
            )
            self._device.set_depth_color_sync_enabled(True)

            self._color_stream.start()
            self._depth_stream.start()
            self._frame_index = 0
            self._is_open = True
        except Exception:
            self.close()
            raise

    def read(self) -> RGBDFrame | None:
        if not self._is_open:
            raise RuntimeError("PrimeSense camera is not open")

        color_frame = self._color_stream.read_frame()
        depth_frame = self._depth_stream.read_frame()

        color_rgb = np.frombuffer(
            color_frame.get_buffer_as_uint8(),
            dtype=np.uint8,
        ).reshape(color_frame.height, color_frame.width, 3)
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        depth_mm = np.frombuffer(
            depth_frame.get_buffer_as_uint16(),
            dtype=np.uint16,
        ).reshape(depth_frame.height, depth_frame.width)

        # Copy both arrays because OpenNI may reuse its frame buffers after this method returns.
        color_bgr = color_bgr.copy()
        depth_mm = depth_mm.copy()

        # OpenNI timestamps are expressed in microseconds.
        timestamp_ms = max(color_frame.timestamp, depth_frame.timestamp) / 1000.0

        frame = RGBDFrame(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
        )

        self._frame_index += 1

        return frame

    def close(self) -> None:
        for stream in (self._color_stream, self._depth_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass

        for stream in (self._color_stream, self._depth_stream):
            if stream is not None:
                try:
                    stream.destroy()
                except Exception:
                    pass

        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass

        if self._openni2 is not None:
            try:
                if self._openni2.is_initialized():
                    self._openni2.unload()
            except Exception:
                pass

        self._color_stream = None
        self._depth_stream = None
        self._device = None
        self._is_open = False
