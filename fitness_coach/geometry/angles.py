"""
Pure geometry utilities for computing joint angles from 2D landmark points.
No MediaPipe or OpenCV dependency here — this module is intentionally
framework-agnostic so it can be unit tested in isolation.
"""
import math
from typing import Tuple

Point = Tuple[float, float]


class AngleCalculator:
    """Computes joint angles using the law of cosines."""

    @staticmethod
    def calculate_angle(a: Point, b: Point, c: Point) -> float:
        """
        Returns the angle at point b (in degrees), formed by the rays b->a and b->c.

        Example: calculate_angle((0,1),(0,0),(1,0)) == 90.0
        because b->a points straight up and b->c points straight right.
        """
        ax, ay = a
        bx, by = b
        cx, cy = c

        # Vectors from the vertex (b) to the other two points
        ba = (ax - bx, ay - by)
        bc = (cx - bx, cy - by)

        # Dot product and magnitudes
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.hypot(*ba)
        mag_bc = math.hypot(*bc)

        if mag_ba == 0 or mag_bc == 0:
            # Degenerate case — landmarks collapsed to the same point.
            # Return 180 (straight/neutral) rather than crashing on div-by-zero.
            return 180.0

        cos_angle = dot / (mag_ba * mag_bc)
        # Clamp for float precision errors that can push cos slightly outside [-1, 1]
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def vertical_deviation(point: Point, ref_top: Point, ref_bottom: Point) -> float:
        """
        Measures how far `point` deviates laterally (x-axis) from the straight
        line connecting ref_top and ref_bottom, in normalized coordinate units.

        Used later for knee valgus detection (Phase 2), but the primitive
        belongs in Phase 1's geometry engine.
        """
        # If the reference line is perfectly vertical, deviation is just the x difference
        if ref_top[1] == ref_bottom[1]:
            return abs(point[0] - ref_top[0])

        # Parametrize the line and find the expected x at point's y-height
        t = (point[1] - ref_top[1]) / (ref_bottom[1] - ref_top[1])
        expected_x = ref_top[0] + t * (ref_bottom[0] - ref_top[0])
        return abs(point[0] - expected_x)