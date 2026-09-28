# JevBench submission preregistration — Linnaeus-0.1.0-2B

Everything in this document is committed before the benchmark run, matching
the operator's preregistration convention (`docs/v1.2-additions-*.md` in
fstandhartinger/jevbench).

## System

- **Name for the board:** Linnaeus-0.1.0-2B
- **Weights:** [`pi-dal/Linnaeus-0.1.0-2B`](https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B) —
  Apache-2.0, rank-8 LoRA adapter + shared candidate-scorer head
  (`adapter.safetensors` + `linnaeus.json`)
- **Base:** `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc` (Apache-2.0)
- **Server:** this repository, `linnaeus-serve` (`src/linnaeus/server.py`) —
  a thin FastAPI wrapper over `linnaeus.predictor.Predictor`, the same
  class used for our published measurements
- **Code licence:** Apache-2.0 (`pi-dal/Linnaeus`)

## Request/response contract

`POST /v1/systemone` with `{"state", "model", "questions": {"decision": {...}}}`
returns `{"answers": {"decision": {...}}, "usage": {"input_tokens": n, "images": 0}, "model": "Linnaeus-0.1.0-2B"}`.

- `choice` → `answers.decision.probabilities` keyed by option label (+ `choice`)
- `noul` → `answers.decision.noul` = P(yes); probabilities `{"yes","no"}` also present
- `score` → `answers.decision.probabilities` keyed by level index (+ `score` expected value)

State, instructions and criteria pass through unchanged. One calibrated
temperature per question type (choice 1.6947, noul 2.9416, score 4.8547 —
fitted on an independent calibration partition, recorded in `linnaeus.json`).

## Endpoint conditions (what we ask the operator to run)

- The author's server (`linnaeus-serve`, commit noted at run time) on the
  operator's GPU; **batch size 1, serial requests, maximum input 4,096 tokens**
- Inputs exceeding 4,096 tokens return HTTP 422 — a refusal counted as wrong,
  consistent with the standing rule; nothing is truncated
- Requests with images are not part of the frozen set; the endpoint rejects
  them with 422 rather than guessing
- LoRA merged at load (score head is a fused vocabulary row — logits index
  `score_row_id` at each `<|fim_suffix|>` marker position)
- Prefix caching across requests: **off** (each decision is a fresh forward;
  our shared-prefix path only deduplicates within a multi-question request)

## Cost basis

Self-hosted open weights: no per-token tariff exists. Like other
local-weight rows we suggest `cost = null` (unmetered; not free) or, if a
hosted-reference estimate is preferred, the OpenRouter `Qwen3.5-2B`-class
input tariff (the 9B-class tariff used for Open-Jev-2B is the conservative
fallback). The model emits no output tokens.

## Training-data overlap statement

Linnaeus-0.1.0-2B trains exclusively on the upstream Dohnuts recipe's frozen
train split (`data/processed/v1`, source hashes recorded in
`runs/2b/recipe.json` and the checkpoint manifest): MultiXI/Bespoke-style
typed-decision rows, VQAv2, typed-decisions — **no JevBench items are in the
training or calibration partitions**. The checkpoint predates any contact
with the benchmark's sealed items: we ran only the 231-item public set.
We welcome the operator's normalized exact-match audit against our published
data projection.

## Published self-measurement (public items only)

- JevBench v1.2.2 public set (231 tasks): **73.16%** overall —
  easy 100%, standard ~82%, hard 49.6% (per-family record in
  `runs/2b/jevbench/results.jsonl`, versioned in this repo)
- Held-out development selection: seed 42, update 2,800 of 3,600
  (max dev macro accuracy; test labels never selected)
- Serial, no retries, native distributions — identical to the recorded run
