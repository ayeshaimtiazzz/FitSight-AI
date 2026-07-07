from fitness_coach.geometry.angles import AngleCalculator


def test_right_angle():
    # b->a points up, b->c points right => 90 degrees
    angle = AngleCalculator.calculate_angle((0, 1), (0, 0), (1, 0))
    assert round(angle, 1) == 90.0


def test_straight_line():
    # a, b, c collinear and b between a and c => 180 degrees
    angle = AngleCalculator.calculate_angle((-1, 0), (0, 0), (1, 0))
    assert round(angle, 1) == 180.0


def test_acute_angle():
    angle = AngleCalculator.calculate_angle((1, 1), (0, 0), (1, 0))
    assert 40 < angle < 50  # should be 45 degrees


def test_degenerate_case_does_not_crash():
    angle = AngleCalculator.calculate_angle((0, 0), (0, 0), (1, 0))
    assert angle == 180.0

def test_vertical_deviation_no_deviation():
    # Point sits exactly on the straight line between top and bottom
    deviation = AngleCalculator.vertical_deviation(
        point=(0.5, 0.6), ref_top=(0.5, 0.4), ref_bottom=(0.5, 0.8)
    )
    assert deviation == 0.0


def test_vertical_deviation_measures_lateral_offset():
    # Point is 0.15 units to the left of the straight line
    deviation = AngleCalculator.vertical_deviation(
        point=(0.35, 0.6), ref_top=(0.5, 0.4), ref_bottom=(0.5, 0.8)
    )
    assert round(deviation, 2) == 0.15


def test_vertical_deviation_handles_horizontal_reference_line():
    # Edge case: ref_top and ref_bottom have the same y — avoids divide-by-zero
    deviation = AngleCalculator.vertical_deviation(
        point=(0.6, 0.5), ref_top=(0.5, 0.5), ref_bottom=(0.8, 0.5)
    )
    assert round(deviation, 2) == 0.10

def test_3d_right_angle():
    angle = AngleCalculator.calculate_angle_3d((0, 1, 0), (0, 0, 0), (1, 0, 0))
    assert round(angle, 1) == 90.0


def test_3d_depth_movement_does_not_fake_a_bend():
    # Arm fully extended, but the wrist moves toward the camera (z changes).
    # A correct 3D angle should stay close to 180 despite the z shift —
    # this is exactly the false-positive case the fix addresses.
    angle = AngleCalculator.calculate_angle_3d(
        a=(0, -1, 0),      # shoulder
        b=(0, 0, 0),       # elbow
        c=(0, 1, -0.3),    # wrist, same straight-line direction but closer to camera
    )
    assert angle > 150  # still reads as nearly straight, not falsely bent