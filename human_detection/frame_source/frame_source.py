from __future__ import annotations
from abc import ABC, abstractmethod
from types import TracebackType

from .rgbd_frame import RGBDFrame


class FrameSource(ABC):
    """Common interface for frame sources used by the tracking pipeline."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Return the frame width in pixels."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Return the frame height in pixels."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Return the source frame rate in frames per second."""

    @property
    @abstractmethod
    def frame_count(self) -> int | None:
        """Return the total frame count, or None for an unbounded source."""

    @abstractmethod
    def open(self) -> None:
        """Open the source and prepare it for frame reads."""

    @abstractmethod
    def read(self) -> RGBDFrame | None:
        """Return the next frame, or None when no more frames are available."""

    @abstractmethod
    def close(self) -> None:
        """Stop frame reads and release the source."""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
