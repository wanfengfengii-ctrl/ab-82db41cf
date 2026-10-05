"""Unit tests for request validation and the interpolation pipeline."""

import math

import pytest

from app.interpolation import (
    MAX_QUERIES,
    MAX_SAMPLES,
    RequestError,
    interpolate,
    prepare_request,
)

SQRT1_2 = math.sqrt(0.5)


def sample(t, q):
    return {"t_ns": t, "q": q}


def rotation_z(theta):
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def payload(times, quats, queries, max_gap=1000):
    return {
        "samples": [sample(t, q) for t, q in zip(times, quats)],
        "queries": queries,
        "max_interval_ns": max_gap,
    }


def error_types(exc):
    return {e["type"] for e in exc.errors}


def error_indices(exc):
    return {e.get("index") for e in exc.errors if e.get("index") is not None}


# ---------------------------------------------------------------- happy path

def test_basic_interpolation_midpoint():
    body = payload(
        [0, 100],
        [[1, 0, 0, 0], [SQRT1_2, 0, 0, SQRT1_2]],
        [0, 50, 100],
    )
    out = interpolate(prepare_request(body))
    assert len(out) == 3
    expected = rotation_z(math.pi / 4.0)
    for got, want in zip(out[1], expected):
        assert abs(got - want) < 1e-12
    # Endpoints reproduce the samples (canonical sign applied afterwards).
    assert out[0] == [1.0, 0.0, 0.0, 0.0]
    for got, want in zip(out[2], [SQRT1_2, 0.0, 0.0, SQRT1_2]):
        assert abs(got - want) < 1e-15


def test_non_unit_quaternions_are_treated_as_rotations():
    body = payload(
        [0, 100],
        # 90 deg about z (norm 5), written as non-unit components.
        [[5, 0, 0, 0], [5 * SQRT1_2, 0, 0, 5 * SQRT1_2]],
        [50],
    )
    out = interpolate(prepare_request(body))
    expected = rotation_z(math.pi / 4.0)
    # First non-zero component positive canonical sign for expected too.
    for got, want in zip(out[0], expected):
        assert abs(got - want) < 1e-12


def test_sign_flip_input_keeps_arc_continuous():
    # The second sample is the SAME rotation as +90deg but written with the
    # negative-equivalent quaternion. The pipeline must unflip it.
    body = payload(
        [0, 100],
        [[1, 0, 0, 0], [-SQRT1_2, 0, 0, -SQRT1_2]],
        [25, 50, 75],
    )
    out = interpolate(prepare_request(body))
    expected_angles = [math.pi / 8.0, math.pi / 4.0, 3 * math.pi / 8.0]
    for q, theta in zip(out, expected_angles):
        for got, want in zip(q, rotation_z(theta)):
            assert abs(got - want) < 1e-12


def test_outputs_are_unit_and_sign_continuous():
    times = list(range(0, 500, 50))
    quats = []
    for k, t in enumerate(times):
        q = rotation_z(0.1 * k)
        if k % 2 == 1:
            q = [-c for c in q]  # randomize equivalent signs in the input
        quats.append(q)
    queries = list(range(0, 451, 25))
    body = payload(times, quats, queries, max_gap=100)
    out = interpolate(prepare_request(body))

    # First result: first non-zero component positive.
    first_nonzero = next(c for c in out[0] if c != 0.0)
    assert first_nonzero > 0.0

    prev = out[0]
    for q in out:
        assert abs(math.sqrt(sum(c * c for c in q)) - 1.0) < 1e-12
        # Inner product of identical rotations is >= 0 by sign choice.
        assert sum(a * b for a, b in zip(prev, q)) >= -1e-12
        prev = q


def test_endpoint_query_with_interval_equal_to_limit_is_allowed():
    body = payload([0, 100], [[1, 0, 0, 0], rotation_z(0.5)], [0, 100], 100)
    out = interpolate(prepare_request(body))
    assert len(out) == 2


# --------------------------------------------------------------- rejections

def reject(body):
    with pytest.raises(RequestError) as exc:
        prepare_request(body)
    return exc.value


def test_zero_quaternion_rejected_with_index():
    body = payload([0, 100, 200], [[1, 0, 0, 0], [0, 0, 0, 0],
                                   rotation_z(0.3)], [50])
    exc = reject(body)
    assert "zero_quaternion" in error_types(exc)
    assert 1 in error_indices(exc)


def test_non_finite_component_rejected_with_index():
    body = payload([0, 100], [[1, 0, 0, float("nan")], rotation_z(0.3)], [50])
    exc = reject(body)
    assert "non_finite" in error_types(exc)
    assert 0 in error_indices(exc)


def test_non_increasing_sample_times_rejected_with_index():
    body = payload([0, 100, 100], [[1, 0, 0, 0], rotation_z(0.2),
                                   rotation_z(0.3)], [50])
    exc = reject(body)
    assert "non_increasing_time" in error_types(exc)
    assert 2 in error_indices(exc)


def test_non_increasing_queries_rejected_with_index():
    body = payload([0, 100], [[1, 0, 0, 0], rotation_z(0.3)], [50, 50])
    exc = reject(body)
    assert "non_increasing_time" in error_types(exc)
    assert 1 in error_indices(exc)


def test_gap_too_large_rejected_with_index():
    body = payload([0, 100, 300],
                   [[1, 0, 0, 0], rotation_z(0.2), rotation_z(0.4)],
                   [150], max_gap=150)
    exc = reject(body)
    # Interval 200 -> 300 (200 ns) exceeds 150; index of right sample is 2.
    assert "gap_too_large" in error_types(exc)
    assert 2 in error_indices(exc)


def test_query_outside_span_rejected_with_index():
    body = payload([0, 100], [[1, 0, 0, 0], rotation_z(0.3)], [150])
    exc = reject(body)
    assert "out_of_range" in error_types(exc)
    assert 0 in error_indices(exc)


def test_180_degree_adjacent_rotation_rejected():
    # Identity and 180-degree rotation about x: zero absolute dot product.
    body = payload([0, 100], [[1, 0, 0, 0], [0, 1, 0, 0]], [50])
    exc = reject(body)
    assert "ambiguous_rotation" in error_types(exc)
    assert 1 in error_indices(exc)


def test_near_180_degree_rotation_rejected():
    almost = [math.cos((math.pi - 1e-13) / 2), math.sin((math.pi - 1e-13) / 2),
              0.0, 0.0]
    body = payload([0, 100], [[1, 0, 0, 0], almost], [50])
    exc = reject(body)
    assert "ambiguous_rotation" in error_types(exc)


def test_sample_count_bounds():
    too_few = payload([0], [[1, 0, 0, 0]], [0])
    assert "out_of_bounds" in error_types(reject(too_few))

    times = [float(t) for t in range(MAX_SAMPLES + 1)]
    quats = [rotation_z(0.01 * k) for k in range(len(times))]
    too_many = payload(times, quats, [0.0], max_gap=1.0)
    assert "out_of_bounds" in error_types(reject(too_many))


def test_query_count_bounds():
    ok_pair = [[1, 0, 0, 0], rotation_z(0.3)]
    none = payload([0, 100], ok_pair, [])
    assert "out_of_bounds" in error_types(reject(none))

    many = [float(k) for k in range(MAX_QUERIES + 1)]
    too_many = payload([0, 10000], ok_pair, many, max_gap=10000)
    assert "out_of_bounds" in error_types(reject(too_many))


def test_invalid_max_interval():
    body = payload([0, 100], [[1, 0, 0, 0], rotation_z(0.3)], [50], 0)
    assert "invalid_type" in error_types(reject(body))


def test_quaternion_wrong_dimension_rejected():
    body = {
        "samples": [sample(0, [1, 0, 0]), sample(100, rotation_z(0.3))],
        "queries": [50],
        "max_interval_ns": 1000,
    }
    assert "invalid_type" in error_types(reject(body))


def test_multiple_errors_all_located():
    # Zero quaternion at index 1 AND a gap overflow at index 2: nothing is
    # returned until every located problem is fixed.
    body = payload([0, 100, 400],
                   [[1, 0, 0, 0], [0, 0, 0, 0], rotation_z(0.4)],
                   [50], max_gap=200)
    exc = reject(body)
    types = error_types(exc)
    assert "zero_quaternion" in types
    assert "gap_too_large" in types


def test_epoch_nanosecond_integers_keep_exact_gaps():
    # Epoch ns (~1.7e18) exceed 2**53, where float64 spacing is 256 ns.
    # Integer timestamps must keep exact 1,000,000 ns gaps against an
    # exactly equal max interval, instead of being rejected by float noise.
    epoch_ns = 1_700_000_000_000_000_000
    times = [epoch_ns + 1_000_000 * k for k in range(5)]
    quats = [rotation_z(0.1 * k) for k in range(5)]
    queries = [epoch_ns + 2_500_000]  # exactly 3/4 between samples 2 and 3
    body = payload(times, quats, queries, max_gap=1_000_000)
    out = interpolate(prepare_request(body))
    for got, want in zip(out[0], rotation_z(0.25)):
        assert abs(got - want) < 1e-12


def test_integer_and_fractional_times_can_mix():
    body = payload([0, 100], [[1, 0, 0, 0], rotation_z(0.4)], [25.0])
    out = interpolate(prepare_request(body))
    for got, want in zip(out[0], rotation_z(0.1)):
        assert abs(got - want) < 1e-12
