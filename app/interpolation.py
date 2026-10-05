"""Request validation and shortest-arc quaternion interpolation.

The validation is intentionally hand-rolled (instead of declared via a
pydantic model) so that every rejection can carry the precise index of the
offending sample/query and so that no partial result is ever produced:
either the whole request is accepted or a single structured 4xx response
describing every located problem is returned.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

from .quaternion import dot, normalize, slerp

# JSON numbers are either int (exact, arbitrary precision in Python) or
# float; timestamps keep the int representation for nanosecond accuracy.
Number = Union[int, float]

MIN_SAMPLES = 2
MAX_SAMPLES = 2000
MIN_QUERIES = 1
MAX_QUERIES = 500

# Rotations whose angle is 180 degrees have a zero absolute inner product of
# their unit quaternions.  At exactly 180 degrees infinitely many shortest
# arcs exist (genuine sign ambiguity), and slerp is ill-conditioned in the
# neighbourhood, so anything within this angular guard band is rejected.
# Normalization/rounding noise is at the 1e-16 level; 1e-12 is conservative.
ANTIPODAL_EPS = 1e-12

_QUATERNION_DIM = 4


class RequestError(ValueError):
    """Collects all located validation errors for one request."""

    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


@dataclass(frozen=True)
class AttitudeRequest:
    times: list[Number]
    quaternions: list[list[float]]  # unit quaternions, sign-continuous
    queries: list[Number]


def _is_real_number(value: Any) -> bool:
    # bool is a subclass of int; a timestamp/interval must not be boolean.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_time(value: Any, errors, etype: str, loc: str,
                index: Optional[int]) -> Optional[Number]:
    """Return a timestamp/interval preserving exact integer precision.

    Nanosecond timestamps routinely exceed 2**53 (epoch ns is ~1.7e18),
    where float64 spacing is hundreds of nanoseconds; integer JSON numbers
    are therefore kept as Python ints so gap arithmetic and ordering stay
    exact.  Fractional values (and float inputs) stay float.
    """
    if isinstance(value, bool) or not _is_real_number(value):
        _err(errors, etype, loc, "must be a finite number", index)
        return None
    if isinstance(value, int):
        return value  # type: ignore[return-value]
    if not math.isfinite(value):
        _err(errors, etype, loc, "must be a finite number", index)
        return None
    return float(value)


def _err(errors: list[dict[str, Any]], etype: str, loc: str,
         message: str, index: Optional[int] = None) -> None:
    item: dict[str, Any] = {"type": etype, "loc": loc, "message": message}
    if index is not None:
        item["index"] = index
    errors.append(item)


def _validate(payload: Any) -> tuple[list[Number], list[list[float]],
                                     list[Number], Number]:
    """Validate the whole request, collecting every located error.

    Samples are kept in index-aligned arrays with ``None`` placeholders for
    structurally invalid entries, so independent checks (time ordering, gap
    limits, the antipodal rule) can still run on the remaining data instead
    of masking one error with another.
    """
    errors: list[dict[str, Any]] = []

    if not isinstance(payload, dict):
        raise RequestError([{
            "type": "invalid_body",
            "loc": "$",
            "message": "request body must be a JSON object",
        }])

    raw_samples = payload.get("samples")
    raw_queries = payload.get("queries")
    raw_max_gap = payload.get("max_interval_ns")

    # ---- samples -----------------------------------------------------------
    times: list[Optional[Number]] = []
    quats: list[Optional[list[float]]] = []

    if not isinstance(raw_samples, list):
        _err(errors, "missing_or_invalid", "samples",
             "field 'samples' is required and must be an array")
    else:
        count = len(raw_samples)
        if not (MIN_SAMPLES <= count <= MAX_SAMPLES):
            _err(errors, "out_of_bounds", "samples",
                 f"expected between {MIN_SAMPLES} and {MAX_SAMPLES} samples, "
                 f"got {count}")
        for i, sample_ in enumerate(
                raw_samples if MIN_SAMPLES <= count <= MAX_SAMPLES else ()):
            loc = f"samples[{i}]"
            t_value: Optional[Number] = None
            q_value: Optional[list[float]] = None

            if not isinstance(sample_, dict):
                _err(errors, "invalid_type", loc,
                     "sample must be an object", i)
            else:
                t = sample_.get("t_ns")
                t_value = _parse_time(t, errors, "non_finite",
                                      f"{loc}.t_ns", i)

                q = sample_.get("q")
                qloc = f"{loc}.q"
                if not isinstance(q, list) or len(q) != _QUATERNION_DIM:
                    _err(errors, "invalid_type", qloc,
                         "q must be an array of exactly 4 numbers "
                         "[w, x, y, z]", i)
                else:
                    components: list[float] = []
                    all_finite = True
                    for k, component in enumerate(q):
                        if (not _is_real_number(component)
                                or not math.isfinite(component)):
                            _err(errors, "non_finite", f"{qloc}[{k}]",
                                 "every quaternion component must be a finite "
                                 "number", i)
                            all_finite = False
                        else:
                            components.append(float(component))
                    if all_finite:
                        # math.hypot stays well-defined for extreme magnitudes.
                        n = math.hypot(*components)
                        if n == 0.0:
                            _err(errors, "zero_quaternion", qloc,
                                 "quaternion must be non-zero", i)
                        else:
                            q_value = normalize(components)

            times.append(t_value)
            quats.append(q_value)

    # ---- queries -----------------------------------------------------------
    queries: list[Optional[Number]] = []
    if not isinstance(raw_queries, list):
        _err(errors, "missing_or_invalid", "queries",
             "field 'queries' is required and must be an array")
    else:
        count = len(raw_queries)
        if not (MIN_QUERIES <= count <= MAX_QUERIES):
            _err(errors, "out_of_bounds", "queries",
                 f"expected between {MIN_QUERIES} and {MAX_QUERIES} queries, "
                 f"got {count}")
        for j, query in enumerate(
                raw_queries if MIN_QUERIES <= count <= MAX_QUERIES else ()):
            queries.append(_parse_time(query, errors, "non_finite",
                                       f"queries[{j}]", j))

    # ---- max interval ------------------------------------------------------
    max_gap = _parse_time(raw_max_gap, errors, "invalid_type",
                          "max_interval_ns", None)
    if max_gap is None:
        max_gap = math.inf
    elif max_gap <= 0:
        _err(errors, "invalid_type", "max_interval_ns",
             "max_interval_ns must be a finite positive number")
        max_gap = math.inf

    if not isinstance(raw_samples, list) or not isinstance(raw_queries, list):
        # Arrays are missing entirely; aligned checks below cannot run.
        raise RequestError(errors)

    times_ok = all(t is not None for t in times)
    time_axis_ok = times_ok

    # ---- strictly increasing sample times ---------------------------------
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        if t0 is not None and t1 is not None and t1 <= t0:
            _err(errors, "non_increasing_time", f"samples[{i}].t_ns",
                 f"sample times must be strictly increasing; t_ns at index "
                 f"{i} ({t1!r}) is not greater than at index {i - 1} "
                 f"({t0!r})", i)
            time_axis_ok = False

    # ---- strictly increasing query times ----------------------------------
    query_axis_ok = True
    for j in range(1, len(queries)):
        q0, q1 = queries[j - 1], queries[j]
        if q0 is not None and q1 is not None and q1 <= q0:
            _err(errors, "non_increasing_time", f"queries[{j}]",
                 f"query times must be strictly increasing; query at index "
                 f"{j} ({q1!r}) is not greater than at index {j - 1} "
                 f"({q0!r})", j)
            query_axis_ok = False

    # ---- per-interval rules (need a well-formed time axis) ----------------
    if time_axis_ok and len(times) >= 2:
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]  # type: ignore[operator]
            if gap > max_gap:
                _err(errors, "gap_too_large",
                     f"samples[{i - 1}].t_ns -> samples[{i}].t_ns",
                     f"sample interval {gap:g} ns exceeds max_interval_ns "
                     f"{max_gap:g} ns", i)

        for i in range(1, len(quats)):
            q0, q1 = quats[i - 1], quats[i]
            if q0 is None or q1 is None:
                continue  # a structural error for this pair was already logged
            # Absolute inner product: rotations ignore quaternion sign.
            if abs(dot(q0, q1)) <= ANTIPODAL_EPS:
                _err(errors, "ambiguous_rotation",
                     f"samples[{i - 1}].q -> samples[{i}].q",
                     "adjacent rotations are 180 degrees apart; the shortest "
                     "arc is not unique", i)

    # ---- query coverage (endpoints allowed, enclosing gap <= limit) -------
    if time_axis_ok and query_axis_ok and len(times) >= 2:
        t_first = times[0]
        t_last = times[-1]
        for j, qt in enumerate(queries):
            if qt is None:
                continue
            if qt < t_first or qt > t_last:  # type: ignore[operator]
                _err(errors, "out_of_range", f"queries[{j}]",
                     f"query time {qt:g} ns lies outside the sample span "
                     f"[{t_first:g}, {t_last:g}] ns", j)

    if errors:
        raise RequestError(errors)

    return ([t for t in times if t is not None],
            [q for q in quats if q is not None],
            [q for q in queries if q is not None],
            max_gap)


def prepare_request(payload: Any) -> AttitudeRequest:
    times, quats, queries, _max_gap = _validate(payload)

    # Resolve the +/- equivalence of quaternions into one sign-continuous
    # chain BEFORE interpolation, so slerp never crosses the far arc.
    chain = [quats[0]]
    for i in range(1, len(quats)):
        q = quats[i]
        if dot(chain[-1], q) < 0.0:
            q = [-c for c in q]
        chain.append(q)

    return AttitudeRequest(times=times, quaternions=chain, queries=queries)


def interpolate(req: AttitudeRequest) -> list[list[float]]:
    """Slerp every query time along the shortest arc.

    Query times equal to a sample time reproduce that sample exactly;
    otherwise the query is bracketed by its two adjacent samples.
    """
    results: list[list[float]] = []
    times = req.times

    for qt in req.queries:
        idx = bisect.bisect_left(times, qt)
        if idx < len(times) and times[idx] == qt:
            result = list(req.quaternions[idx])
        else:
            # qt is strictly inside (times[idx - 1], times[idx]); endpoints of
            # the sample span were handled by the equality branch above.
            left = idx - 1
            right = idx
            t0, t1 = times[left], times[right]
            tau = (qt - t0) / (t1 - t0)
            result = slerp(req.quaternions[left], req.quaternions[right], tau)

        results.append(result)

    # Canonical sign for presentation: the first non-zero component of the
    # first result is positive, and every later result keeps the sign whose
    # inner product with the previous result is non-negative.
    first = results[0]
    for component in first:
        if component != 0.0:
            if component < 0.0:
                results[0] = [-c for c in first]
            break

    for k in range(1, len(results)):
        if dot(results[k - 1], results[k]) < 0.0:
            results[k] = [-c for c in results[k]]

    return results
