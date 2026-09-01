import subprocess
from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Write tracking frames to MP4 and finalize them as H.264."""

    def __init__(self, output_path: str | Path, fps: float) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.output_path.with_name(
            f"{self.output_path.stem}.mp4v-temp.mp4"
        )
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.temporary_path),
                fourcc,
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(
                    f"Unable to create temporary video: {self.temporary_path}"
                )

        self._writer.write(frame)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def finalize(self) -> None:
        self.close()
        if self._frame_count == 0:
            raise RuntimeError(f"No frames written to video: {self.output_path}")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(self.temporary_path),
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
                str(self.output_path),
            ],
            check=True,
        )
        self.temporary_path.unlink()
