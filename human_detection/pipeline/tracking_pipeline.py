from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from human_detection.bbox_trajectory import TargetBBoxTrajectory
from human_detection.frame_source import FrameSource
from human_detection.person_tracker import PersonTracker
from human_detection.pose_estimator import PoseEstimator
from human_detection.position_estimator import PositionEstimator
from human_detection.target_tracker import TargetTracker
from human_detection.visualization import (
    VideoWriter,
    render_debug_frame,
    render_tracking_frame,
)

from .tracking_frame_result import TrackingFrameResult


@dataclass(frozen=True)
class TrackingPipelineConfig:
    """Output and display settings for the tracking pipeline."""

    bbox_enabled: bool
    bbox_path: str | Path

    video_enabled: bool
    video_path: str | Path

    debug_enabled: bool
    debug_path: str | Path

    display_enabled: bool
    display_window_name: str = "Target Tracking"


class TrackingPipeline:
    """Run target tracking and manage optional outputs and display."""

    def __init__(
        self,
        person_tracker: PersonTracker,
        target_tracker: TargetTracker,
        target_bbox_trajectory: TargetBBoxTrajectory,
        pose_estimator: PoseEstimator,
        position_estimator: PositionEstimator,
        config: TrackingPipelineConfig,
    ) -> None:
        self.person_tracker = person_tracker
        self.target_tracker = target_tracker
        self.target_bbox_trajectory = target_bbox_trajectory
        self.pose_estimator = pose_estimator
        self.position_estimator = position_estimator
        self.config = config

    def run(self, source: FrameSource) -> None:
        tracking_writer = None
        if self.config.video_enabled:
            tracking_writer = VideoWriter(
                output_path=self.config.video_path,
                fps=source.fps,
            )

        debug_writer = None
        if self.config.debug_enabled:
            debug_writer = VideoWriter(
                output_path=self.config.debug_path,
                fps=source.fps,
            )

        progress_bar = tqdm(
            total=source.frame_count,
            desc="Tracking",
            unit="frame",
        )
        display_window_opened = False
        processed_count = 0

        self.target_tracker.reset()
        self.target_bbox_trajectory.reset()

        try:
            with source:
                while True:
                    rgbd_frame = source.read()
                    if rgbd_frame is None:
                        break

                    frame_height, frame_width = rgbd_frame.color_bgr.shape[:2]

                    # Track all persons in the frame.
                    tracked_persons = self.person_tracker.track(rgbd_frame.color_bgr)

                    # Find target.
                    target_result = self.target_tracker.track(
                        frame_bgr=rgbd_frame.color_bgr,
                        tracked_persons=tracked_persons,
                        frame_index=rgbd_frame.frame_index,
                    )

                    # Update the target bbox trajectory.
                    trajectory_entry = self.target_bbox_trajectory.update(
                        frame_index=rgbd_frame.frame_index,
                        target=target_result.target,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )

                    target_bbox = (
                        target_result.target.bbox
                        if target_result.target is not None
                        else None
                    )

                    pose_result = self.pose_estimator.estimate(
                        frame_bgr=rgbd_frame.color_bgr,
                        bbox=target_bbox,
                    )

                    position_mm = self.position_estimator.estimate(
                        depth_mm=rgbd_frame.depth_mm,
                        bbox=target_bbox,
                    )

                    progress_bar.update(1)
                    processed_count += 1

                    result = TrackingFrameResult(
                        rgbd_frame=rgbd_frame,
                        tracked_persons=tracked_persons,
                        target_result=target_result,
                        trajectory_entry=trajectory_entry,
                        pose=pose_result,
                        position_mm=position_mm,
                    )

                    tracking_frame = None
                    if tracking_writer is not None or self.config.display_enabled:
                        tracking_frame = render_tracking_frame(result)

                    if tracking_writer is not None:
                        tracking_writer.write(tracking_frame)

                    # save debug video
                    if debug_writer is not None:
                        debug_frame = render_debug_frame(result)
                        debug_writer.write(debug_frame)

                    # display online
                    if self.config.display_enabled:
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
            self.target_bbox_trajectory.save(
                self.config.bbox_path,
                source,
            )
            print(f"Bounding boxes saved to: {self.config.bbox_path}")
