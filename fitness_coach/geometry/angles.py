"""
Pure geometry utilities for computing joint angles from landmark points.
No MediaPipe or OpenCV dependency here — this module is intentionally
framework-agnostic so it can be unit tested in isolation.
"""
import math
from typing import Tuple

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


class AngleCalculator:
    """Computes joint angles using the law of cosines, in 2D and 3D."""

    @staticmethod
    def calculate_angle(a: Point2D, b: Point2D, c: Point2D) -> float:
        """
        Returns the angle at point b (in degrees), formed by the rays b->a and b->c,
        using flat (x, y) coordinates only.

        Example: calculate_angle((0,1),(0,0),(1,0)) == 90.0
        """
        ax, ay = a
        bx, by = b
        cx, cy = c

        ba = (ax - bx, ay - by)
        bc = (cx - bx, cy - by)

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.hypot(*ba)
        mag_bc = math.hypot(*bc)

        if mag_ba == 0 or mag_bc == 0:
            return 180.0

        cos_angle = dot / (mag_ba * mag_bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def calculate_angle_3d(a: Point3D, b: Point3D, c: Point3D) -> float:
        """
        Same as calculate_angle, but uses full 3D (x, y, z) points instead of
        flattened (x, y). This avoids false angle changes caused by movement
        toward/away from the camera (depth), which a pure 2D angle cannot
        distinguish from actual joint flexion.

        a, b, c are (x, y, z) tuples. Returns the angle at b in degrees.
        """
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = c

        ba = (ax - bx, ay - by, az - bz)
        bc = (cx - bx, cy - by, cz - bz)

        dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)

        if mag_ba == 0 or mag_bc == 0:
            return 180.0

        cos_angle = dot / (mag_ba * mag_bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def vertical_deviation(point: Point2D, ref_top: Point2D, ref_bottom: Point2D) -> float:
        """
        Measures how far `point` deviates laterally (x-axis) from the straight
        line connecting ref_top and ref_bottom, in normalized coordinate units.
        Used for knee valgus detection.
        """
        if ref_top[1] == ref_bottom[1]:
            return abs(point[0] - ref_top[0])

        t = (point[1] - ref_top[1]) / (ref_bottom[1] - ref_top[1])
        expected_x = ref_top[0] + t * (ref_bottom[0] - ref_top[0])
        return abs(point[0] - expected_x)