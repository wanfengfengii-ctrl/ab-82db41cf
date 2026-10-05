"""API-level tests using FastAPI's in-process test client."""

import math

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SQRT1_2 = math.sqrt(0.5)


def sample(t, q):
    return {"t_ns": t, "q": q}


def rotation_z(theta):
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_interpolate_happy_path():
    body = {
        "samples": [
            sample(0, [1, 0, 0, 0]),
            sample(100, [SQRT1_2, 0, 0, SQRT1_2]),
        ],
        "queries": [0, 25, 50, 75, 100],
        "max_interval_ns": 100,
    }
    r = client.post("/api/attitudes/interpolate", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    out = data["quaternions"]
    assert len(out) == 5

    angles = [0.0, math.pi / 8.0, math.pi / 4.0,
              3 * math.pi / 8.0, math.pi / 2.0]
    for q, theta in zip(out, angles):
        assert abs(math.sqrt(sum(c * c for c in q)) - 1.0) <= 1e-12
        for got, want in zip(q, rotation_z(theta)):
            assert abs(got - want) <= 1e-9

    # Canonical first sign and continuous sign chain.
    assert next(c for c in out[0] if c != 0.0) > 0.0
    for a, b in zip(out, out[1:]):
        assert sum(x * y for x, y in zip(a, b)) >= -1e-12


def test_equivalent_negative_sign_does_not_flip_the_arc():
    body = {
        "samples": [
            sample(0, [1, 0, 0, 0]),
            sample(100, [-SQRT1_2, 0, 0, -SQRT1_2]),  # same rotation, -q
        ],
        "queries": [50],
        "max_interval_ns": 100,
    }
    r = client.post("/api/attitudes/interpolate", json=body)
    assert r.status_code == 200, r.text
    for got, want in zip(r.json()["quaternions"][0], rotation_z(math.pi / 4)):
        assert abs(got - want) <= 1e-9


def _expect_422(body, error_type, index=None):
    r = client.post("/api/attitudes/interpolate", json=body)
    assert r.status_code == 422, r.text
    data = r.json()
    assert "quaternions" not in data, "errors must not produce partial output"
    details = data["details"]
    assert any(e["type"] == error_type for e in details)
    if index is not None:
        assert any(e.get("index") == index for e in details)


def test_gap_too_large_rejected_with_index():
    body = {
        "samples": [
            sample(0, [1, 0, 0, 0]),
            sample(100, rotation_z(0.2)),
            sample(300, rotation_z(0.4)),
        ],
        "queries": [50],
        "max_interval_ns": 150,
    }
    _expect_422(body, "gap_too_large", index=2)


def test_zero_quaternion_rejected_with_index():
    body = {
        "samples": [sample(0, [1, 0, 0, 0]), sample(100, [0, 0, 0, 0])],
        "queries": [50],
        "max_interval_ns": 100,
    }
    _expect_422(body, "zero_quaternion", index=1)


def test_non_increasing_times_rejected_with_index():
    body = {
        "samples": [
            sample(100, [1, 0, 0, 0]),
            sample(100, rotation_z(0.3)),
        ],
        "queries": [100],
        "max_interval_ns": 100,
    }
    _expect_422(body, "non_increasing_time", index=1)


def test_180_degree_rotation_rejected_with_index():
    body = {
        "samples": [sample(0, [1, 0, 0, 0]), sample(100, [0, 1, 0, 0])],
        "queries": [50],
        "max_interval_ns": 100,
    }
    _expect_422(body, "ambiguous_rotation", index=1)


def test_malformed_json_is_400():
    r = client.post("/api/attitudes/interpolate",
                    content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_json"
