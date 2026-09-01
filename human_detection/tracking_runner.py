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


class TrackingRunner:
    """Run target tracking and manage optional outputs and display."""

    def __init__(
        self,
        target_tracker: TargetTracker,
    ) -> None:
        self.target_tracker = target_tracker

    def run(
        self,
        source: FrameSource,
        tracking_writer: VideoWriter | None = None,
        debug_writer: VideoWriter | None = None,
        bbox_output_path: str | Path | None = None,
        display_enabled: bool = False,
        display_window_name: str = "Target Tracking",
    ) -> None:
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
                if display_enabled:
                    if tracking_frame is None:
                        tracking_frame = render_tracking_frame(result)

                    cv2.imshow(display_window_name, tracking_frame)
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
                    cv2.destroyWindow(display_window_name)
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

        if bbox_output_path is not None:
            self.target_tracker.target_bbox_trajectory.save(bbox_output_path, source)
            print(f"Bounding boxes saved to: {bbox_output_path}")
