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