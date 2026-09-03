import numpy as np


class PositionEstimator:
    def __init__(self, intrinsics):
        # intrinsics = [[fx, 0, cx],
        #               [0, fy, cy],
        #               [0, 0, 1]]
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)

    def deproject(
        self,
        depth_mm: float,
        pixel: tuple[float, float],
    ) -> tuple[float, float, float]:
        """
        Deproject a pixel coordinate (u, v) with depth value (Z) to 3D coordinates (X, Y, Z).

        Args:
            depth_mm: Depth value in millimeters.
            pixel: Pixel coordinates (u, v).

        Returns:
            (X, Y, Z): 3D coordinates in millimeters.
        """

        u, v = pixel

        fx = float(self.intrinsics[0, 0])
        fy = float(self.intrinsics[1, 1])
        cx = float(self.intrinsics[0, 2])
        cy = float(self.intrinsics[1, 2])

        Z = float(depth_mm)

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return X, Y, Z

    def estimate_depth(
        self,
        depth_mm: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> float | None:
        """
        Estimate the depth value (Z) from a bounding box in the depth image.


        """
        # TODO: how to define depth value from bbox?
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        # region of interest (ROI) for depth estimation
        roi_x1 = int(x1 + width * 0.25)
        roi_x2 = int(x2 - width * 0.25)
        roi_y1 = int(y1 + height * 0.25)
        roi_y2 = int(y2 - height * 0.25)

        roi = depth_mm[roi_y1:roi_y2, roi_x1:roi_x2]

        valid = roi[(roi > 10) & (roi < 10000)]  # 1cm-10m

        if valid.size == 0:
            return None

        depth = float(np.median(valid))

        return depth

    def estimate(
        self,
        depth_mm: np.ndarray | None,
        bbox: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        if depth_mm is None or bbox is None:
            return None

        Z = self.estimate_depth(depth_mm, bbox)
        if Z is None:
            return None

        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        X, Y, Z = self.deproject(Z, (center_x, center_y))

        return X, Y, Z
