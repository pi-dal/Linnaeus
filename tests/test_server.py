"""The serve endpoint must start and answer — not stringify its route signature.

Regression guard for the v1.4 submission: `Request` is imported inside
`create_app` so fastapi stays an optional `serve` extra, which means the route
annotation must be evaluated eagerly. With `from __future__ import annotations`
active, FastAPI resolved "Request" against module globals, where it does not
exist, so the app either failed to start or bound `request` as a required query
parameter and answered 422 to every request.

Requests go through the real ASGI stack (dependency solving included) without
loading weights or starting a socket.
"""

import asyncio
import json

import pytest

from linnaeus import server

QUESTIONS = {"route": {"type": "noul", "instructions": "Is this a refund request?"}}


class _StubPredictor:
    """Stands in for the calibrated predictor; no GPU, weights, or Hub access."""

    def predict(self, state, questions):
        assert set(questions) == set(QUESTIONS)
        answers = {
            qid: {"type": "noul", "noul": 0.75, "confidence": 0.75, "probabilities": {}}
            for qid in questions
        }
        return {"model": "linnaeus", "answers": answers, "usage": {"input_tokens": 11, "images": 0}}


@pytest.fixture
def app(monkeypatch):
    def fake_from_checkpoint(cls, directory, **kwargs):
        return _StubPredictor()

    monkeypatch.setattr(server.Predictor, "from_checkpoint", classmethod(fake_from_checkpoint))
    return server.create_app()


def _post(app, body):
    payload = json.dumps(body).encode()
    inbox = [{"type": "http.request", "body": payload, "more_body": False}]
    sent = []

    async def receive():
        return inbox.pop(0) if inbox else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/systemone",
        "raw_path": b"/v1/systemone",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 51000),
        "server": ("linnaeus", 80),
        "state": {},
    }
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(raw)


def test_answer_carries_model_id_and_latency(app):
    status, body = _post(app, {"state": {"message": "please refund"}, "questions": QUESTIONS})
    assert status == 200
    assert body["model"] == server.DEFAULT_MODEL_ID
    assert body["answers"]["route"]["noul"] == 0.75
    assert body["usage"] == {"input_tokens": 11, "images": 0}
    assert body["latency_s"] >= 0


def test_over_budget_input_is_a_refusal_not_an_outage(app, monkeypatch):
    def refuse(self, state, questions):
        raise ValueError("Input exceeds the token budget")

    monkeypatch.setattr(_StubPredictor, "predict", refuse)
    status, body = _post(app, {"state": {"message": "x" * 10}, "questions": QUESTIONS})
    assert status == 422
    assert "token budget" in body["error"]


def test_malformed_body_is_rejected_before_reading_state(app):
    assert _post(app, {"state": {"message": "hi"}})[0] == 400
    assert _post(app, {"questions": QUESTIONS})[0] == 400


def test_unusable_image_state_is_a_refusal(app):
    status, body = _post(app, {"state": {"image": 1234}, "questions": QUESTIONS})
    assert status == 422
    assert "base64" in body["error"]
