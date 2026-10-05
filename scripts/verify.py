#!/usr/bin/env python3
"""One-shot verification for the attitude interpolation stack.

Runs (and aggregates with a single process exit code):

  1. build check  - byte-compile every package/test/script file
  2. code tests   - the pytest unit + API suite
  3. API smoke    - against the live service over HTTP:
                      a. health endpoint
                      b. a valid interpolation request (norms, accuracy,
                         canonical/continuous signs)
                      c. an over-long gap request (must be a 4xx that names
                         the offending sample index)

Exit code is 0 only when every step passes, otherwise 1.  The service base
URL is taken from $BASE_URL (default http://web:8000, matching compose).
"""

from __future__ import annotations

import json
import math
import os
import py_compile
import subprocess
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://web:8000")
TOL = 1e-9

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


# ------------------------------------------------------------------ 1. build

def check_build() -> None:
    targets = ["app", "tests", "scripts"]
    try:
        py_compile.compile("app/main.py", doraise=True)
        for directory in targets:
            for root, _dirs, files in os.walk(directory):
                for fname in files:
                    if fname.endswith(".py"):
                        py_compile.compile(os.path.join(root, fname),
                                           doraise=True)
        record("build (byte-compile)", True)
    except py_compile.PyCompileError as exc:
        record("build (byte-compile)", False, str(exc))


# ------------------------------------------------------------------ 2. tests

def check_unit_tests() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True, text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else "no pytest output"
    record("code tests (pytest)", proc.returncode == 0, summary)


# ----------------------------------------------------------------- 3. smoke

def _get(path: str):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as resp:
        return resp.status, resp.read()


def _post(path: str, body: dict):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def rotation_z(theta: float) -> list[float]:
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def check_health() -> None:
    try:
        status, body = _get("/health")
        ok = status == 200 and json.loads(body).get("status") == "ok"
        record("smoke: health endpoint", ok, f"HTTP {status}")
    except Exception as exc:  # network errors are a smoke failure
        record("smoke: health endpoint", False, repr(exc))


def check_interpolation() -> None:
    s = math.sqrt(0.5)
    body = {
        "samples": [
            {"t_ns": 0, "q": [1, 0, 0, 0]},
            {"t_ns": 100, "q": [-s, 0, 0, -s]},  # -q equivalent of +90 deg
        ],
        "queries": [0, 25, 50, 75, 100],
        "max_interval_ns": 100,
    }
    try:
        status, data = _post("/api/attitudes/interpolate", body)
        if status != 200:
            record("smoke: valid interpolation", False,
                   f"HTTP {status}: {data}")
            return
        out = data["quaternions"]
        expected = [rotation_z(a) for a in
                    (0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8,
                     math.pi / 2)]

        problems = []
        if len(out) != 5:
            problems.append(f"expected 5 results, got {len(out)}")
        for k, (got, want) in enumerate(zip(out, expected)):
            norm = math.sqrt(sum(c * c for c in got))
            if abs(norm - 1.0) > TOL:
                problems.append(f"result {k} not unit (|q|={norm})")
            if max(abs(a - b) for a, b in zip(got, want)) > TOL:
                problems.append(f"result {k} deviates from slerp {got}")
        # First non-zero component of the first row must be positive...
        first_sign = next(c for c in out[0] if c != 0.0)
        if first_sign <= 0:
            problems.append("first result has negative canonical sign")
        # ...and every later row keeps a non-negative inner product.
        for k in range(1, len(out)):
            d = sum(a * b for a, b in zip(out[k - 1], out[k]))
            if d < -TOL:
                problems.append(f"sign flip between results {k - 1} and {k}")

        record("smoke: valid interpolation", not problems,
               "; ".join(problems) or "5 continuous unit quaternions, tol 1e-9")
    except Exception as exc:
        record("smoke: valid interpolation", False, repr(exc))


def check_gap_rejection() -> None:
    body = {
        "samples": [
            {"t_ns": 0, "q": [1, 0, 0, 0]},
            {"t_ns": 100, "q": rotation_z(0.2)},
            {"t_ns": 300, "q": rotation_z(0.4)},  # 200 ns gap, limit 150
        ],
        "queries": [50],
        "max_interval_ns": 150,
    }
    try:
        status, data = _post("/api/attitudes/interpolate", body)
        if not (400 <= status < 500):
            record("smoke: gap rejection", False,
                   f"expected 4xx, got HTTP {status}")
            return
        details = data.get("details", [])
        located = any(e.get("type") == "gap_too_large"
                      and e.get("index") == 2 for e in details)
        no_partial = "quaternions" not in data
        ok = located and no_partial
        record("smoke: over-long gap rejected (index located, no partial)",
               ok, f"HTTP {status}, details={details}")
    except Exception as exc:
        record("smoke: over-long gap rejected", False, repr(exc))


def main() -> int:
    check_build()
    check_unit_tests()
    check_health()
    check_interpolation()
    check_gap_rejection()

    width = max(len(name) for name, _ok, _d in results)
    print(f"\nverify report (service: {BASE_URL})")
    print("-" * (width + 18))
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {name.ljust(width)}"
        if detail and not ok:
            line += f"  -> {detail}"
        elif detail:
            line += f"  ({detail})"
        print(line)
    print("-" * (width + 18))

    passed = sum(1 for _n, ok, _d in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
