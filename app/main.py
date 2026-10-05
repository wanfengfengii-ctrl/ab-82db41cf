"""HTTP layer for the attitude interpolation service.

Endpoints:
  GET  /health                       liveness/readiness probe
  POST /api/attitudes/interpolate    shortest-arc quaternion slerp
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .interpolation import RequestError, interpolate, prepare_request

app = FastAPI(
    title="Aerial Attitude Interpolation Service",
    version="1.0.0",
    description="Projects low-frequency INS attitudes onto camera exposure "
                "times via shortest-arc unit-quaternion slerp.",
)


@app.exception_handler(RequestError)
async def request_error_handler(_: Request, exc: RequestError) -> JSONResponse:
    # 422 is the natural status for well-formed-but-invalid requests; every
    # error entry locates the offending sample/query by index.
    return JSONResponse(status_code=422, content={
        "error": "validation_failed",
        "count": len(exc.errors),
        "details": exc.errors,
    })


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/attitudes/interpolate")
async def interpolate_attitudes(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={
            "error": "invalid_json",
            "count": 1,
            "details": [{
                "type": "invalid_json",
                "loc": "$",
                "message": "request body is not valid JSON",
            }],
        })

    req = prepare_request(payload)
    # Validation above either raises or leaves a fully valid request, so
    # interpolation runs for all queries or for none.
    results = interpolate(req)
    return JSONResponse(content={"quaternions": results})
