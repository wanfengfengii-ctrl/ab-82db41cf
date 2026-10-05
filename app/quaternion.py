"""Quaternion math and slerp interpolation.

Quaternions are represented as plain ``[w, x, y, z]`` lists/tuples.
Every quaternion accepted or produced by this module is a *unit*
quaternion (rotation quaternion).
"""

from __future__ import annotations

import math
from typing import Sequence

# A dot product this close to +/-1 makes the angle numerically zero.
_COS_EPS = 1e-12
# sin(theta) below this means the two quaternions are effectively parallel
# (or antiparallel, which is rejected upstream); linear interpolation is then
# as accurate as the trig formula and avoids a division by a tiny sine.
_SIN_EPS = 1e-12


def norm(q: Sequence[float]) -> float:
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def normalize(q: Sequence[float]) -> list[float]:
    n = norm(q)
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> list[float]:
    """Spherical linear interpolation along the shortest arc.

    ``q0`` and ``q1`` must already be unit quaternions with a non-negative
    inner product, so the interpolation follows the short path (angle in
    ``[0, pi/2]``). ``t`` is in ``[0, 1]``.
    """
    d = dot(q0, q1)
    if d > 1.0:
        d = 1.0
    elif d < -1.0:
        d = -1.0

    if t == 0.0:
        return list(q0)
    if t == 1.0:
        return list(q1)

    if d >= 1.0 - _COS_EPS:
        # Effectively the same rotation.
        result = list(q0)
    elif 1.0 - d * d <= _SIN_EPS * _SIN_EPS:
        # Nearly coincident directions: LERP is numerically safer.
        result = [
            q0[i] + (q1[i] - q0[i]) * t for i in range(4)
        ]
        result = normalize(result)
    else:
        theta = math.acos(d)
        sin_theta = math.sin(theta)
        w0 = math.sin((1.0 - t) * theta) / sin_theta
        w1 = math.sin(t * theta) / sin_theta
        result = [w0 * q0[i] + w1 * q1[i] for i in range(4)]

    # Renormalize to cancel rounding drift, then return exactly.
    n = norm(result)
    return [c / n for c in result]
