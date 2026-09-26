# Linnaeus-0.1.0-2B: JevBench public evaluation

Checkpoint: seed 42, step 2800; SHA-256 `0424555feea3126ee02d8d1b12636fad8cf491a5964a00a04b2fec67bcc0f602`.
JevBench v1.2.2, pinned commit `e105a48f8cdb7f3babb3594424f73e5d7bdc97b9`.

All 231 public tasks are attempted. The official suite has 534 tasks; 303 are unavailable.
This result has no official leaderboard rank or four-axis score. Local compute has no provider tariff.
Overall accuracy: 169/231 (73.16%), including input rejections as wrong answers.

| Public tier | Tasks | Accuracy | Valid distributions | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 72 | 91.67% | 72 | 0.2042 | 0.1604 |
| easy | 48 | 100.00% | 48 | 0.0028 | 0.0171 |
| hard | 111 | 49.55% | 111 | 0.6108 | 0.1141 |

Inputs above the training length: 36; correct 15; maximum input length 3902 tokens.
Their latency is p50 66.84 ms and p95 6019.57 ms.

Public standard latency: p50 46.20 ms, p95 51.00 ms.
Inputs rejected by the deployed API: 0; these count as wrong.
Brier and ECE above cover valid distributions only; rejected inputs have no probability distribution.
Model loading: 4.21 s; first decision: 801.57 ms.
Whole-board peak VRAM during evaluation: 5.96 GiB, including runtime and desktop.

## Quality on identical public tasks

Reference outcomes come from the upstream public per-task artifact, not from a new local run.

| System | Attempted / 231 | Easy | Standard | Hard | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| classifier.dev (fast tier) | 231 | 100.00% | 98.61% | 70.27% | 85.28% |
| DeepSeek V4.1 Flash (thinking default) | 231 | 100.00% | 98.61% | 96.40% | 97.84% |
| djev (Maisa, diffusion-gemma) | 231 | 100.00% | 98.61% | 67.57% | 83.98% |
| Gemini 3.1 Flash-Lite | 231 | 100.00% | 98.61% | 73.87% | 87.01% |
| GLiNER2 (Fastino, gliner2.5-base) | 231 | 97.92% | 63.89% | 36.94% | 58.01% |
| GPT-5.6 Luna (low reasoning effort) | 231 | 100.00% | 97.22% | 96.40% | 97.40% |
| jeff (Logan Markewich, GLiFormer 400M) | 231 | 100.00% | 75.00% | 38.74% | 62.77% |
| Jev 1.13.0 (TypeSafe AI) | 231 | 100.00% | 98.61% | 72.97% | 86.58% |
| Laya (Convai Innovations, ModernBERT-large 421M) | 231 | 95.83% | 69.44% | 35.14% | 58.44% |
| Linnaeus-0.1.0-2B (seed 42) | 231 | 100.00% | 91.67% | 49.55% | 73.16% |
| Needle 3 (Cactus, 2-bit, local CPU) | 164 | 47.92% | 16.67% | 15.32% | 22.51% |
| Needle 3, options as tools (post-hoc adapter mode) | 120 | 66.67% | 26.39% | 0.00% | 22.08% |
| Bespoke Nimble 9B (Bespoke Labs) | 231 | 100.00% | 93.06% | 36.94% | 67.53% |
| open-alternative-jev (Qwen3.5-4B, IkerMoel) | 231 | 100.00% | 83.33% | 56.76% | 74.03% |
| open-jev-deberta-v3-large (local CPU) | 231 | 100.00% | 43.06% | 37.84% | 52.38% |
| OpenJev (DiffusionGemma 26B-A4B NVFP4, razorback16) | 231 | 100.00% | 97.22% | 63.96% | 81.82% |
| openjev-sglang (Qwen3.6-35B-A3B on SGLang) | 231 | 100.00% | 94.44% | 72.97% | 85.28% |
| openJev Verdict (heman10x, ModernBERT-base 151M) | 231 | 85.42% | 62.50% | 37.84% | 55.41% |
| Qwen3.8 27B (Chutes TEE) | 171 | 100.00% | 98.61% | 42.34% | 71.86% |
| SemIf, formerly OpenJev (Qwen3.5-4B, TheoLeeCJ) | 231 | 100.00% | 98.61% | 61.26% | 80.95% |
| system-one-open (Gemma 4 E2B LoRA on an L4) | 231 | 100.00% | 93.06% | 48.65% | 73.16% |
| system-one (Qwen3-8B, Sean Goedecke) | 231 | 100.00% | 88.89% | 48.65% | 71.86% |

## Measurement scope

- 231 public tasks; 303 private or non-redistributed tasks are unavailable, including the entire judge tier.
- Official runner and scoring functions; one decision per call, no concurrency, no retries.
- State, instructions and criteria are unchanged; labels and rationales are never given as answers.
- Checkpoint, temperatures and the 4096-token serving limit are frozen before this evaluation; the training budget remains 2048 tokens.
- Input rejections count as wrong; no truncation or exclusion of over-budget questions.
- No provider tariff exists for local GPU execution. Cost and the four-axis composite stay null.
- Reference quality is recomputed on identical public IDs; published full-suite scores are not comparable.
- Public-subset Intelligence renormalizes the available easy/standard/hard weights; it is not leaderboard Intelligence.
- Latency is local end-to-end wall time. Model loading is separate; first decision is retained.
- The official speed cohort has 242 standard/judge tasks; only its 72 public standard tasks are available.
