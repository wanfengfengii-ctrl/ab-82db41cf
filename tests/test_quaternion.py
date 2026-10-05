"""Unit tests for quaternion slerp."""

import math

from app.quaternion import dot, normalize, norm, slerp

SQRT1_2 = math.sqrt(0.5)


def rotation_z(theta: float) -> list[float]:
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def test_normalize_non_unit_input():
    q = normalize([2.0, 0.0, 0.0, 0.0])
    assert q == [1.0, 0.0, 0.0, 0.0]
    assert abs(norm(q) - 1.0) < 1e-15


def test_slerp_endpoints_reproduce_inputs():
    q0 = normalize([1.0, 2.0, 3.0, 4.0])
    q1 = normalize([4.0, 3.0, 2.0, 1.0])
    if dot(q0, q1) < 0:
        q1 = [-c for c in q1]
    assert slerp(q0, q1, 0.0) == q0
    assert slerp(q0, q1, 1.0) == q1


def test_slerp_midpoint_half_angle_about_z():
    q0 = rotation_z(0.0)
    q1 = rotation_z(math.pi / 2.0)
    mid = slerp(q0, q1, 0.5)
    expected = rotation_z(math.pi / 4.0)
    for got, want in zip(mid, expected):
        assert abs(got - want) < 1e-12


def test_slerp_constant_speed_on_symmetric_arc():
    # Equal-time slerp samples lie on the great-circle arc; consecutive dot
    # products must be equal and the product of halves must meet in the middle.
    q0 = rotation_z(0.0)
    q1 = rotation_z(math.pi / 3.0)
    a = slerp(q0, q1, 1.0 / 3.0)
    b = slerp(q0, q1, 2.0 / 3.0)
    assert abs(dot(q0, a) - dot(a, b)) < 1e-12
    assert abs(dot(a, b) - dot(b, q1)) < 1e-12


def test_slerp_result_is_unit_norm():
    q0 = normalize([1.0, 1.0, 0.0, 0.0])
    q1 = normalize([1.0, 0.0, 1.0, 0.0])
    for k in range(11):
        q = slerp(q0, q1, k / 10.0)
        assert abs(norm(q) - 1.0) < 1e-15


def test_slerp_identical_quaternions():
    q = normalize([0.3, 0.4, 0.5, 0.6])
    assert slerp(q, q, 0.7) == q
