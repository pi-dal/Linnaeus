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
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import] (macOS-only dependency)
import numpy as np

from linnaeus.predictor import render, render_question

IMAGE_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>\n"


class MlxPredictor:
    """predict() parity with linnaeus.predictor.Predictor, MLX backend."""

    def __init__(
        self, model_path: str | Path, runtime: str | Path | None = None, vision: bool | None = None
    ):
        from transformers import AutoTokenizer

        self.model_path = Path(model_path)
        runtime_path = Path(runtime) if runtime else self._find_runtime()
        contract = json.loads(runtime_path.read_text())
        self.score_row = contract["score_row_id"]
        self.temperatures = contract["temperatures"]
        self.max_length = contract.get("max_length", 2048)
        tok = AutoTokenizer.from_pretrained(self.model_path)
        assert tok is not None
        self.tokenizer = tok
        self.marker_id = self.tokenizer.convert_tokens_to_ids(contract["marker"])

        # A VLM build is detected by the presence of vision weights or the
        # `vision` flag; it uses mlx_vlm's processor for pixel_values.
        self.vision = vision
        if self.vision is None:
            self.vision = (
                any(self.model_path.glob("**/model.visual*"))
                or "vlm" in self.model_path.name.lower()
            )
        if self.vision:
            from mlx_vlm import load as vlm_load  # ty: ignore[unresolved-import]

            self.model, self.processor = vlm_load(self.model_path)
        else:
            from mlx_lm import load as lm_load  # ty: ignore[unresolved-import]

            self.model, _ = lm_load(self.model_path)
            self.processor = None

    def _find_runtime(self) -> Path:
        for base in (self.model_path, self.model_path.parent / "merged-hf"):
            p = base / "linnaeus-runtime.json"
            if p.exists():
                return p
        raise FileNotFoundError("linnaeus-runtime.json not found beside the model")

    def _forward(self, ids, cache=None, image=None):
        if image is None:
            out = self.model(mx.array(list(ids))[None], cache=cache)
            return out.logits if hasattr(out, "logits") else out
        assert self.processor is not None
        inp = self.processor(text=[ids], images=[image])  # ids here is text
        tok_ids = inp["input_ids"]
        extra = {k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values")}
        pixels = inp["pixel_values"]
        pixels = pixels if isinstance(pixels, mx.array) else mx.array(np.array(pixels))
        out = self.model(input_ids=mx.array(tok_ids), pixel_values=pixels, cache=cache, **extra)
        return out.logits if hasattr(out, "logits") else out

    def _score_text(self, text: str, image=None):
        """Whole-sequence fallback path (no shared prefix)."""
        if image is None:
            ids = self.tokenizer(text)["input_ids"]
            if len(ids) > self.max_length:
                ids = ids[-self.max_length :]
            positions = [i for i, t in enumerate(ids) if t == self.marker_id]
            logits = self._forward(ids)
        else:
            if not self.vision:
                raise ValueError("This MLX build has no vision tower; use a -vlm- export")
            formatted = IMAGE_PREFIX + text.removeprefix(IMAGE_PREFIX)
            assert self.processor is not None
            inp = self.processor(text=[formatted], images=[image])
            flat = inp["input_ids"]
            flat = flat[0] if getattr(flat, "ndim", 1) > 1 else flat
            flat = flat.tolist() if hasattr(flat, "tolist") else list(flat)
            positions = [i for i, t in enumerate(flat) if t == self.marker_id]
            logits = self._forward(formatted, image=image)
        return logits[0, positions, self.score_row].astype(mx.float32)

    def _predict_cached(self, state_text: str, questions, image):
        """Shared-prefix path: forward the state once, branch per question.

        Questions share `State: ...\n` (+ image tokens) as a common prompt
        prefix. We forward it once into the KV/delta caches, snapshot the
        cache state, and each question forwards only its own suffix.
        Falls back per question when the token boundary is not clean.
        Numeric note: delta-rule chunked updates produce ~1e-1 logit drift
        vs full-forward (same class as the platform gap); argmax decisions
        are unaffected. Measured: 1.4x faster on 8 text questions, 4.5x on
        6 image questions (vision encoding runs once).
        """
        from mlx_lm.models.cache import make_prompt_cache  # ty: ignore[unresolved-import]

        prefix_text = f"State: {state_text}\n"
        if image is not None:
            # processor expands image tokens at inference; the token boundary
            # check below operates on raw text ids, which is consistent
            prefix_text = IMAGE_PREFIX + prefix_text
        pids = self.tokenizer(prefix_text)["input_ids"]

        answers = {}
        # VLM wrapper lacks make_cache; the language model owns the hybrid
        # (ArraysCache + KVCache) cache list.
        lm = getattr(self.model, "language_model", self.model)
        cache = make_prompt_cache(lm)
        self._forward(pids if image is None else prefix_text, cache=cache, image=image)
        snapshot = [c.state for c in cache]
        prefix_len = len(pids)

        for qid, question in questions.items():
            content, labels = render_question(state_text, question, has_image=image is not None)
            # render_question already prepends IMAGE_PREFIX when has_image
            full_ids = self.tokenizer(content)["input_ids"]
            if len(full_ids) > self.max_length:
                scores = self._score_text(content, image)
            elif full_ids[:prefix_len] != pids or all(
                t != self.marker_id for t in full_ids[prefix_len:]
            ):
                scores = self._score_text(content, image)
            else:
                for c, s in zip(cache, snapshot):
                    c.state = s
                suffix = full_ids[prefix_len:]
                logits = self._forward(suffix, cache=cache)
                positions = [i for i, t in enumerate(suffix) if t == self.marker_id]
                scores = logits[0, positions, self.score_row].astype(mx.float32)
            answers[qid] = (labels, scores)
        return answers

    def predict(self, state, questions):
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        image = None
        if isinstance(state, Mapping):
            image = state.get("image")
            state = {k: v for k, v in state.items() if k != "image"}
        state_text = render(state)
        answers = {}
        try:
            scored = self._predict_cached(state_text, questions, image)
        except Exception as e:
            import warnings

            warnings.warn(
                f"shared-prefix path failed ({e}); falling back to per-question forward",
                stacklevel=2,
            )
            scored = {
                qid: (
                    render_question(state_text, q, has_image=image is not None)[1],
                    self._score_text(
                        render_question(state_text, q, has_image=image is not None)[0], image
                    ),
                )
                for qid, q in questions.items()
            }
        for qid, question in questions.items():
            labels, scores = scored[qid]
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
