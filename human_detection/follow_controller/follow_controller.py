import math

import numpy as np

from human_detection.utils import Point3D
from .robot_velocity import VelocityCommand, VelocityFeedback


class FollowController:
    """Convert a camera-frame human position in millimeters into a bounded velocity command."""

    def __init__(
        self,
        camera_to_robot_rotation: np.ndarray | None,
        camera_to_robot_translation_mm: np.ndarray | None,
        desired_distance_m: float,
        gain_x: float,
        gain_y: float,
        gain_yaw: float,
        max_positive_x_speed_mps: float,
        max_negative_x_speed_mps: float,
        max_y_speed_mps: float,
        max_yaw_rate_radps: float,
        max_x_accel_mps2: float,
        max_y_accel_mps2: float,
        max_yaw_accel_radps2: float,
        x_deadband_m: float,
        y_deadband_m: float,
        yaw_deadband_rad: float,
        position_filter_alpha: float,
        velocity_filter_alpha: float = 0.35,
        max_target_speed_mps: float = 1.5,
        timeout_threshold: float = 1.0,
    ) -> None:
        # Extrinsics map camera coordinates to robot coordinates, in millimeters.
        self.camera_to_robot_rotation = (
            None
            if camera_to_robot_rotation is None
            else np.array(camera_to_robot_rotation, dtype=np.float64, copy=True)
        )
        self.camera_to_robot_translation_mm = (
            None
            if camera_to_robot_translation_mm is None
            else np.array(camera_to_robot_translation_mm, dtype=np.float64, copy=True)
        )

        self.desired_distance_m = desired_distance_m

        self.gain_x = gain_x
        self.gain_y = gain_y
        self.gain_yaw = gain_yaw

        # speed limit
        self.max_positive_x_speed_mps = max_positive_x_speed_mps
        self.max_negative_x_speed_mps = max_negative_x_speed_mps
        self.max_y_speed_mps = max_y_speed_mps
        self.max_yaw_rate_radps = max_yaw_rate_radps

        self.max_x_accel_mps2 = max_x_accel_mps2
        self.max_y_accel_mps2 = max_y_accel_mps2
        self.max_yaw_accel_radps2 = max_yaw_accel_radps2

        # deadband
        self.x_deadband_m = x_deadband_m
        self.y_deadband_m = y_deadband_m
        self.yaw_deadband_rad = yaw_deadband_rad

        # filter alpha
        self.position_filter_alpha = position_filter_alpha
        self.velocity_filter_alpha = velocity_filter_alpha

        self.max_target_speed_mps = max_target_speed_mps

        self.timeout_threshold = timeout_threshold  # seconds

        # state
        self.last_valid_timestamp_ms: float | None = None
        self.last_command = np.zeros(3, dtype=np.float64)
        self.last_human_position: np.ndarray | None = None
        self.last_human_velocity = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.last_valid_timestamp_ms = None
        self.last_human_position = None
        self.last_command.fill(0.0)
        self.last_human_velocity.fill(0.0)

    def update(
        self,
        human_position_camera: Point3D | None,
        velocity_feedback: VelocityFeedback | None,
        timestamp_ms: float,
    ) -> VelocityCommand:
        if (
            velocity_feedback is None
            or self.camera_to_robot_rotation is None
            or self.camera_to_robot_translation_mm is None
        ):
            return VelocityCommand(0.0, 0.0, 0.0, valid=False)

        if self.last_valid_timestamp_ms is not None:
            elapsed = (timestamp_ms - self.last_valid_timestamp_ms) / 1000.0
            if elapsed >= self.timeout_threshold:
                self.reset()

        if human_position_camera is None:
            if self.last_valid_timestamp_ms is None:
                return VelocityCommand(0.0, 0.0, 0.0, valid=False)

            return VelocityCommand(
                vx=float(self.last_command[0]),
                vy=float(self.last_command[1]),
                wz=float(self.last_command[2]),
                valid=True,
            )

        human_position_robot = human_position_camera.transform(
            self.camera_to_robot_rotation,
            self.camera_to_robot_translation_mm,
        )
        # Use robot-frame horizontal coordinates in meters.
        human_position = (
            np.asarray(
                [human_position_robot.x, human_position_robot.y], dtype=np.float64
            )
            / 1000.0
        )

        if self.last_valid_timestamp_ms is None:
            self.last_valid_timestamp_ms = timestamp_ms
            self.last_human_position = human_position.copy()
            return VelocityCommand(0.0, 0.0, 0.0, valid=True)

        dt = (timestamp_ms - self.last_valid_timestamp_ms) / 1000.0

        # Filter position.
        human_position = self.filter(
            human_position, self.last_human_position, self.position_filter_alpha
        )

        # Estimate velocity.
        human_velocity = self.estimate_human_velocity(
            human_position=human_position,
            velocity_feedback=velocity_feedback,
            dt=dt,
        )
        human_velocity = self.filter(
            human_velocity, self.last_human_velocity, self.velocity_filter_alpha
        )

        x, y = human_position[:2]

        heading_error = math.atan2(y, x)
        forward_error = x - self.desired_distance_m

        last_vx, last_vy, last_wz = self.last_command
        max_vx_change = self.max_x_accel_mps2 * dt
        max_vy_change = self.max_y_accel_mps2 * dt
        max_wz_change = self.max_yaw_accel_radps2 * dt

        # wz = K_yaw * atan2(y, x)
        raw_wz = (
            0.0
            if abs(heading_error) <= self.yaw_deadband_rad
            else self.gain_yaw * heading_error
        )
        clipped_wz = float(
            np.clip(raw_wz, -self.max_yaw_rate_radps, self.max_yaw_rate_radps)
        )  # limit velocity
        wz = float(
            np.clip(
                clipped_wz,
                last_wz - max_wz_change,
                last_wz + max_wz_change,
            )
        )  # limit acceleration

        # v_cmd = v_h - wz_cmd * J * p + K * (p - p_target)

        # vx = v_hx + wz_cmd * y + K_x * (x - x_target)
        correction_x = (
            0.0
            if abs(forward_error) <= self.x_deadband_m
            else self.gain_x * forward_error
        )
        raw_vx = float(human_velocity[0]) + wz * y + correction_x
        clipped_vx = float(
            np.clip(
                raw_vx,
                -self.max_negative_x_speed_mps,
                self.max_positive_x_speed_mps,
            )
        )
        vx = float(
            np.clip(
                clipped_vx,
                last_vx - max_vx_change,
                last_vx + max_vx_change,
            )
        )

        # vy_raw = v_hy - wz_cmd * x + K_y * (y - y_target)
        correction_y = 0.0 if abs(y) <= self.y_deadband_m else self.gain_y * y
        raw_vy = float(human_velocity[1]) - wz * x + correction_y
        clipped_vy = float(
            np.clip(
                raw_vy,
                -self.max_y_speed_mps,
                self.max_y_speed_mps,
            )
        )
        vy = float(
            np.clip(
                clipped_vy,
                last_vy - max_vy_change,
                last_vy + max_vy_change,
            )
        )

        command = np.asarray([vx, vy, wz], dtype=np.float64)

        self.last_valid_timestamp_ms = timestamp_ms
        self.last_command = command
        self.last_human_velocity = human_velocity.copy()
        self.last_human_position = human_position.copy()

        return VelocityCommand(
            vx=vx,
            vy=vy,
            wz=wz,
            valid=True,
        )

    def estimate_human_velocity(
        self,
        human_position: np.ndarray,
        velocity_feedback: VelocityFeedback,
        dt: float,
    ) -> np.ndarray:
        """
        Estimate target world velocity, expressed in the current body frame.
        """
        # Estimate velocity
        relative_velocity = (human_position[:2] - self.last_human_position[:2]) / dt
        rotation_velocity = velocity_feedback.wz * np.asarray(
            [-human_position[1], human_position[0]]
        )
        human_velocity = (
            relative_velocity
            + np.asarray([velocity_feedback.vx, velocity_feedback.vy])
            + rotation_velocity
        )

        # Limit velocity
        speed = float(np.linalg.norm(human_velocity))
        if speed > self.max_target_speed_mps:
            human_velocity *= self.max_target_speed_mps / speed

        return human_velocity

    def filter(self, current_value: float, last_value: float, alpha: float) -> float:
        return alpha * current_value + (1.0 - alpha) * last_value
