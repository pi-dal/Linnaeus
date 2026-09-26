"""MLX inference path for Apple Silicon (macOS dev + iOS via mlx-swift).

The merged export embeds the scalar decision head as an extra lm_head row
(see scripts/export_merged_hf.py): the score for each candidate is the logit
of `score_row_id` at that candidate's `<|fim_suffix|>` marker position, so a
stock MLX LM forward is all we need — no hidden-state plumbing.

Same predict() contract as Predictor: state + questions -> answers with
probabilities/choice/noul/score, temperature-calibrated per question type.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import mlx.core as mx

from linnaeus.predictor import render, render_question


class MlxPredictor:
    """predict() parity with linnaeus.predictor.Predictor, MLX backend."""

    def __init__(self, model_path: str | Path, runtime: str | Path | None = None):
        from mlx_lm import load
        from transformers import AutoTokenizer

        self.model_path = Path(model_path)
        runtime_path = Path(runtime) if runtime else self._find_runtime()
        contract = json.loads(runtime_path.read_text())
        self.score_row = contract["score_row_id"]
        self.temperatures = contract["temperatures"]
        self.max_length = contract.get("max_length", 2048)
        self.model, _ = load(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.marker_id = self.tokenizer.convert_tokens_to_ids(contract["marker"])

    def _find_runtime(self) -> Path:
        for base in (self.model_path, self.model_path.parent / "merged-hf"):
            p = base / "linnaeus-runtime.json"
            if p.exists():
                return p
        raise FileNotFoundError("linnaeus-runtime.json not found beside the model")

    def predict(self, state, questions):
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        if isinstance(state, Mapping):
            state = {k: v for k, v in state.items() if k != "image"}
            # image questions are out of scope for the text-only MLX path
        state_text = render(state)
        answers = {}
        for qid, question in questions.items():
            content, labels = render_question(state_text, question)
            ids = self.tokenizer(content)["input_ids"]
            if len(ids) > self.max_length:
                ids = ids[-self.max_length :]
            positions = [i for i, t in enumerate(ids) if t == self.marker_id]
            logits = self.model(mx.array(ids)[None])
            scores = logits[0, positions, self.score_row].astype(mx.float32)
            temperature = self.temperatures[question["type"]]
            probabilities = mx.softmax(scores / temperature)
            values = [float(v) for v in probabilities]
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in values)
            answer = {
                "type": question["type"],
                "confidence": max(0.0, min(1.0, 1 - entropy / math.log(len(labels)))),
            }
            if question["type"] == "noul":
                answer["noul"] = values[1]
                answer["confidence"] = max(values)
            else:
                answer["probabilities"] = dict(zip(labels, values))
                if question["type"] == "choice":
                    answer["choice"] = labels[values.index(max(values))]
                else:
                    answer["score"] = sum(i * p for i, p in enumerate(values))
            answers[qid] = answer
        return {"model": "linnaeus-mlx", "answers": answers}
