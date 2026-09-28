"""TypeSafe-compatible HTTP server for Linnaeus.

Exposes the calibrated Predictor behind `POST /v1/systemone`, the wire format
JevBench's TypeSafeAdapter and CLM's system_one client both speak:

    {"state": ..., "model": "...", "questions": {"decision": {...}}}
      -> {"answers": {...}, "usage": {"input_tokens": n, "images": 0},
          "model": "Linnaeus-0.1.0-2B"}

Inputs beyond the token budget return 422 (a refusal, not an outage) —
matching the benchmark protocol. Image questions are rejected the same way:
the published checkpoint is vision-capable but this JSON-only endpoint does
not accept pixel payloads.

Run:
    linnaeus-serve --checkpoint pi-dal/Linnaeus-0.1.0-2B --port 8700
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import time
from typing import Any

from linnaeus.predictor import Predictor

DEFAULT_CHECKPOINT = "pi-dal/Linnaeus-0.1.0-2B"
DEFAULT_MODEL_ID = "Linnaeus-0.1.0-2B"

_predictor: Predictor | None = None
_model_id = DEFAULT_MODEL_ID


def _decode_state(state: Any):
    """JSON state -> predictor state. `image` fields must be base64 PNG/JPEG."""
    if isinstance(state, dict) and "image" in state:
        from PIL import Image

        raw = state["image"]
        if not isinstance(raw, str):
            raise ValueError("state['image'] must be a base64-encoded image")
        payload = raw.split(",", 1)[-1] if raw.startswith("data:") else raw
        state = {**state, "image": Image.open(io.BytesIO(base64.b64decode(payload)))}
    return state


def create_app(checkpoint: str = DEFAULT_CHECKPOINT, model_id: str = DEFAULT_MODEL_ID):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    global _predictor, _model_id
    _predictor = Predictor.from_checkpoint(checkpoint)
    _model_id = model_id

    app = FastAPI(title="Linnaeus", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True, "model": _model_id}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": _model_id, "object": "model"}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        state = body.get("state")
        questions = body.get("questions")
        if state is None or not isinstance(questions, dict) or not questions:
            return JSONResponse(
                {"error": "expected {state, questions: {id: {type, instructions, criteria}}}"},
                status_code=400,
            )
        try:
            state = _decode_state(state)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        started = time.perf_counter()
        try:
            assert _predictor is not None
            out = _predictor.predict(state, questions)
        except ValueError as e:
            # Input over the token budget or malformed question -> refusal.
            return JSONResponse({"error": str(e)}, status_code=422)
        except Exception as e:  # model-side failure -> honest 500
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
        out["model"] = _model_id
        out["latency_s"] = time.perf_counter() - started
        return out

    return app


def main():
    parser = argparse.ArgumentParser(prog="linnaeus-serve")
    parser.add_argument(
        "--checkpoint", default=os.environ.get("LINNAEUS_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--model-id", default=os.environ.get("LINNAEUS_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--host", default=os.environ.get("LINNAEUS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LINNAEUS_PORT", "8700")))
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.checkpoint, args.model_id)
    print(f"[linnaeus-serve] {_model_id} on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
