"""MLX inference path for Apple Silicon (macOS dev + iOS via mlx-swift).

The merged export embeds the scalar decision head as an extra lm_head row
(see scripts/export_merged_hf.py): the score for each candidate is the logit
of `score_row_id` at that candidate's `<|fim_suffix|>` marker position, so a
stock LM forward is all we need — no hidden-state plumbing.

Two weight flavors:
- text-only MLX builds (mlx-lm convert) for decisions without images
- VLM builds (mlx-vlm convert) that keep the vision tower, for state['image']

Same predict() contract as Predictor: state + questions -> answers with
probabilities/choice/noul/score, temperature-calibrated per question type.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import numpy as np

from linnaeus.predictor import render, render_question

IMAGE_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>\n"


class MlxPredictor:
    """predict() parity with linnaeus.predictor.Predictor, MLX backend."""

    def __init__(self, model_path: str | Path, runtime: str | Path | None = None,
                 vision: bool | None = None):
        from transformers import AutoTokenizer

        self.model_path = Path(model_path)
        runtime_path = Path(runtime) if runtime else self._find_runtime()
        contract = json.loads(runtime_path.read_text())
        self.score_row = contract["score_row_id"]
        self.temperatures = contract["temperatures"]
        self.max_length = contract.get("max_length", 2048)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.marker_id = self.tokenizer.convert_tokens_to_ids(contract["marker"])

        # A VLM build is detected by the presence of vision weights or the
        # `vision` flag; it uses mlx_vlm's processor for pixel_values.
        self.vision = vision
        if self.vision is None:
            self.vision = any(self.model_path.glob("**/model.visual*")) or "vlm" in self.model_path.name.lower()
        if self.vision:
            from mlx_vlm import load as vlm_load

            self.model, self.processor = vlm_load(self.model_path)
        else:
            from mlx_lm import load as lm_load

            self.model, _ = lm_load(self.model_path)
            self.processor = None

    def _find_runtime(self) -> Path:
        for base in (self.model_path, self.model_path.parent / "merged-hf"):
            p = base / "linnaeus-runtime.json"
            if p.exists():
                return p
        raise FileNotFoundError("linnaeus-runtime.json not found beside the model")

    def _score(self, text: str, image=None):
        positions = None
        if image is None:
            ids = self.tokenizer(text)["input_ids"]
            if len(ids) > self.max_length:
                ids = ids[-self.max_length :]
            positions = [i for i, t in enumerate(ids) if t == self.marker_id]
            logits = self.model(mx.array(ids)[None])
        else:
            if not self.vision:
                raise ValueError("This MLX build has no vision tower; use a -vlm- export")
            formatted = IMAGE_PREFIX + text.removeprefix(IMAGE_PREFIX)
            inp = self.processor(text=[formatted], images=[image])
            ids = inp["input_ids"]
            ids = ids if isinstance(ids, np.ndarray) else np.array(ids)
            flat = ids[0] if ids.ndim > 1 else ids
            positions = [i for i, t in enumerate(flat.tolist()) if t == self.marker_id]
            extra = {k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values")}
            pixels = inp["pixel_values"]
            pixels = pixels if isinstance(pixels, mx.array) else mx.array(np.array(pixels))
            out = self.model(input_ids=mx.array(ids), pixel_values=pixels, **extra)
            logits = out.logits if hasattr(out, "logits") else out
        return logits[0, positions, self.score_row].astype(mx.float32)

    def predict(self, state, questions):
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        image = None
        if isinstance(state, Mapping):
            image = state.get("image")
            state = {k: v for k, v in state.items() if k != "image"}
        state_text = render(state)
        answers = {}
        for qid, question in questions.items():
            content, labels = render_question(state_text, question, has_image=image is not None)
            scores = self._score(content, image)
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
