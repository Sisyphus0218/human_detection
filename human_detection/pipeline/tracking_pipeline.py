from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from human_detection.bbox_trajectory import TargetBBoxTrajectory
from human_detection.follow_controller.follow_controller import FollowController
from human_detection.follow_controller.robot_velocity import VelocityFeedback
from human_detection.frame_source import FrameSource, RGBDFrame
from human_detection.person_tracker import PersonTracker
from human_detection.pose_estimator import PoseEstimator
from human_detection.position_estimator import PositionEstimator
from human_detection.target_tracker import TargetTracker
from human_detection.visualization import (
    VideoWriter,
    render_tracking_frame,
)

from .tracking_frame_result import TrackingFrameResult


@dataclass(frozen=True)
class TrackingPipelineConfig:
    """Output and display settings for the tracking pipeline."""

    video_enabled: bool
    video_path: str | Path

    display_enabled: bool
    display_window_name: str = "Target Tracking"
    render_bbox: bool = True
    render_pose: bool = True
    render_position: bool = True


class TrackingPipeline:
    """Run target tracking, trajectory updates, pose and position estimation."""

    def __init__(
        self,
        person_tracker: PersonTracker,
        target_tracker: TargetTracker,
        target_bbox_trajectory: TargetBBoxTrajectory,
        pose_estimator: PoseEstimator,
        position_estimator: PositionEstimator,
        config: TrackingPipelineConfig,
        follow_controller: FollowController,
    ) -> None:
        self.person_tracker = person_tracker
        self.target_tracker = target_tracker
        self.target_bbox_trajectory = target_bbox_trajectory
        self.pose_estimator = pose_estimator
        self.position_estimator = position_estimator
        self.config = config
        self.follow_controller = follow_controller

    def run(
        self,
        source: FrameSource,
        velocity_feedback_provider: (
            Callable[[RGBDFrame], VelocityFeedback | None] | None
        ) = None,
    ) -> None:
        """Run tracking; missing feedback produces an invalid zero command."""
        self.follow_controller.reset()
        self.target_tracker.reset()
        self.target_bbox_trajectory.reset()

        progress_bar = tqdm(total=source.frame_count, desc="Tracking", unit="frame")
        processed_count = 0
        writer = None
        window_opened = False

        try:
            with source:
                if self.config.video_enabled:
                    writer = VideoWriter(self.config.video_path, source.fps)
                while True:
                    rgbd_frame = source.read()
                    if rgbd_frame is None:
                        break

                    velocity_feedback = (
                        velocity_feedback_provider(rgbd_frame)
                        if velocity_feedback_provider is not None
                        else None
                    )
                    result = self.process_frame(rgbd_frame, velocity_feedback)

                    if self.config.display_enabled or self.config.video_enabled:
                        rendered = render_tracking_frame(
                            result,
                            render_bbox=self.config.render_bbox,
                            render_pose=self.config.render_pose,
                            render_position=self.config.render_position,
                        )
                        if writer is not None:
                            writer.write(rendered)
                        if self.config.display_enabled:
                            window_opened = True
                            cv2.imshow(self.config.display_window_name, rendered)

                    progress_bar.update(1)
                    processed_count += 1

                    if self.config.display_enabled:
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break

        except KeyboardInterrupt:
            tqdm.write("Tracking interrupted by user")

        finally:
            progress_bar.close()
            try:
                if writer is not None:
                    writer.close()
            finally:
                if window_opened:
                    try:
                        cv2.destroyWindow(self.config.display_window_name)
                    except cv2.error:
                        pass

        if processed_count == 0:
            raise RuntimeError(
                f"No frames received from source: {type(source).__name__}"
            )
        if writer is not None:
            writer.finalize()
            print(f"Tracking video saved to: {writer.output_path}")

    def process_frame(
        self,
        rgbd_frame: RGBDFrame,
        velocity_feedback: VelocityFeedback | None = None,
    ) -> TrackingFrameResult:
        """Track and compute a command; missing feedback produces an invalid zero command."""
        frame_height, frame_width = rgbd_frame.color_bgr.shape[:2]

        # Track all people in the frame.
        tracked_persons = self.person_tracker.track(rgbd_frame.color_bgr)

        # Find target.
        target_bbox = self.target_tracker.track(
            frame_bgr=rgbd_frame.color_bgr,
            tracked_persons=tracked_persons,
            frame_index=rgbd_frame.frame_index,
        )

        # Update the target bbox trajectory.
        self.target_bbox_trajectory.update(
            frame_index=rgbd_frame.frame_index,
            bbox=target_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # Estimate pose.
        pose_result = self.pose_estimator.estimate(
            frame_bgr=rgbd_frame.color_bgr,
            bbox=target_bbox,
        )

        # Estimate position.
        target_position = self.position_estimator.estimate(
            depth_mm=rgbd_frame.depth_mm,
            pose=pose_result,
        )

        # Missing positions must reach the controller for hold/timeout handling.
        velocity_command = self.follow_controller.update(
            human_position_camera=target_position,
            velocity_feedback=velocity_feedback,
            timestamp_ms=rgbd_frame.timestamp_ms,
        )

        return TrackingFrameResult(
            rgbd_frame=rgbd_frame,
            target_bbox=target_bbox,
            pose=pose_result,
            position=target_position,
            velocity_command=velocity_command,
        )
