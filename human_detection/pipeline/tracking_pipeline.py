from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from human_detection.frame_source import FrameSource
from human_detection.target_tracker import TargetTracker
from human_detection.visualization import (
    VideoWriter,
    render_debug_frame,
    render_tracking_frame,
)


@dataclass(frozen=True)
class TrackingPipelineConfig:
    """Output and display settings for the tracking pipeline."""

    bbox_enabled: bool
    bbox_path: str | Path
    tracking_video_enabled: bool
    tracking_video_path: str | Path
    debug_video_enabled: bool
    debug_video_path: str | Path
    display_enabled: bool
    display_window_name: str = "Target Tracking"


class TrackingPipeline:
    """Run target tracking and manage optional outputs and display."""

    def __init__(
        self,
        target_tracker: TargetTracker,
        config: TrackingPipelineConfig,
    ) -> None:
        self.target_tracker = target_tracker
        self.config = config

    def run(self, source: FrameSource) -> None:
        tracking_writer = None
        if self.config.tracking_video_enabled:
            tracking_writer = VideoWriter(
                output_path=self.config.tracking_video_path,
                fps=source.fps,
            )

        debug_writer = None
        if self.config.debug_video_enabled:
            debug_writer = VideoWriter(
                output_path=self.config.debug_video_path,
                fps=source.fps,
            )

        tracking_results = self.target_tracker.track(source)
        progress_bar = tqdm(
            tracking_results,
            total=source.frame_count,
            desc="Tracking",
            unit="frame",
        )
        display_window_opened = False
        processed_count = 0

        try:
            for result in progress_bar:
                processed_count += 1

                tracking_frame = None

                # save tracking video
                if tracking_writer is not None:
                    tracking_frame = render_tracking_frame(result)
                    tracking_writer.write(tracking_frame)

                # save debug video
                if debug_writer is not None:
                    debug_frame = render_debug_frame(result)
                    debug_writer.write(debug_frame)

                # display online
                if self.config.display_enabled:
                    if tracking_frame is None:
                        tracking_frame = render_tracking_frame(result)

                    cv2.imshow(self.config.display_window_name, tracking_frame)
                    display_window_opened = True

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        tqdm.write("Tracking stopped by user")
                        break

        except KeyboardInterrupt:
            tqdm.write("Tracking interrupted by user")

        finally:
            progress_bar.close()
            tracking_results.close()

            if tracking_writer is not None:
                tracking_writer.close()
            if debug_writer is not None:
                debug_writer.close()
            if display_window_opened:
                try:
                    cv2.destroyWindow(self.config.display_window_name)
                except cv2.error:
                    pass

        if processed_count == 0:
            raise RuntimeError(
                f"No frames received from source: {type(source).__name__}"
            )

        if tracking_writer is not None:
            tracking_writer.finalize()
            print(f"Tracking result saved to: {tracking_writer.output_path}")

        if debug_writer is not None:
            debug_writer.finalize()
            print(f"All-tracks debug video saved to: {debug_writer.output_path}")

        if self.config.bbox_enabled:
            self.target_tracker.target_bbox_trajectory.save(
                self.config.bbox_path,
                source,
            )
            print(f"Bounding boxes saved to: {self.config.bbox_path}")
