# Linnaeus

**CUDA-optimized small multimodal decision models.** Forked from an AMD ROCm
research project; the same recipe, retargeted to NVIDIA servers.

The upstream project trains and evaluates on AMD ROCm (RX 7900 XTX). Linnaeus
retargets the same recipe to CUDA:

- torch/torchvision pinned to `+cu128` builds via a dedicated package index
  (`pyproject.toml`); FLA's Triton kernels run unchanged on CUDA
- Triton causal convolution enabled on all backends (upstream gated it to HIP);
  SDPA automatically dispatches to FlashAttention/cuDNN on NVIDIA
- VRAM allocator cap is configurable: `LINNAEUS_VRAM_FRACTION` (default `0.8`;
  use `0.95` on headless servers)
- GPU telemetry auto-detects the DRM device and falls back to NVML
  (`uv sync --extra telemetry`); override with `LINNAEUS_GPU_DEVICE` /
  `LINNAEUS_GPU_INDEX`
- Linux + CUDA only; macOS is not a supported runtime

Toolchain and commands are managed by [mise](https://mise.jdx.dev/)
(`mise.toml`): `mise install` provisions Python 3.12 + uv, `mise tasks` lists
all commands.

```bash
mise install && mise run install
LINNAEUS_VRAM_FRACTION=0.95 uv run python scripts/run_experiment.py
```

On hosts without direct Hugging Face access: `export HF_ENDPOINT=https://hf-mirror.com`.

---

Linnaeus builds small multimodal models for direct decisions. Give the model a
message, a document, or an image, and ask it to choose, judge, or score. It returns
probabilities in a single forward pass, with multiple questions sharing the same
input. That is our take on System 1.

The Python toolkit covers training, inference, evaluation, and agent integration.
The model runs locally on a consumer GPU.

[Linnaeus-0.1.0-2B](https://huggingface.co/pi-dal/Linnaeus-0.1.0-2B) · [Model card](MODEL_CARD.md) · [Documentation](docs/README.md) · [Benchmarks](docs/figures/README.md)

## One message, several decisions

Route a support request and check whether it asks for a refund in the same call:

```python
from linnaeus.predictor import Predictor

model = Predictor.from_checkpoint("pi-dal/Linnaeus-0.1.0-2B")
result = model.predict(
    {"message": "I was charged twice. Please refund the duplicate."},
    {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle the request?",
            "criteria": ["billing", "technical support", "sales"],
        },
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    },
)

print(result["answers"]["route"]["choice"])
print(result["answers"]["refund"]["noul"])
```

Each answer has its own distribution. Use `choice` to select a candidate, `noul`
to estimate whether a statement is true, and `score` to rate something on an
ordered scale. Add a PIL image to the state for visual questions.

See [installation and inference](docs/inference.md) for setup, checkpoint loading,
and response fields. The loader downloads and caches the checkpoint and its
pinned base model, then merges LoRA and applies the saved calibration.

## Train, evaluate, integrate

One workflow prepares the public-data mixture, trains with joint RLCD and
cross-entropy, selects a checkpoint, calibrates it, and runs the evaluations:

```bash
uv run python scripts/run_experiment.py
```

The [training guide](docs/run-experiment.md) covers setup and resuming a run.
The [model card](MODEL_CARD.md) reports quality, latency, and limitations. For
agents, [Bub integration](docs/bub-agent.md) exposes the same interface as one
decision tool through the Bub SDK.

## Benchmarks

Held-out **macro accuracy 80.78%** over 26 task groups (180,031 test examples),
temperature-calibrated on an independent partition. Same recipe, same harness,
same data as the upstream baseline — only the backbone changed
(`Qwen/Qwen3.5-0.8B` → `Qwen/Qwen3.5-2B`, revision pinned).

| Model | Backbone | Test macro accuracy | vs 26 groups |
| --- | --- | ---: | --- |
| **Linnaeus-0.1.0-2B** | Qwen3.5-2B | **80.78%** | 20 wins / 6 ties / 0 losses |
| Dohnuts-0.1.0-0.8B (upstream) | Qwen3.5-0.8B | 78.21% | — |

<details><summary>Per-group accuracy (test split, calibrated)</summary>

| Group | Linnaeus-2B | Dohnuts-0.8B | Δ |
| --- | ---: | ---: | ---: |
| scienceqa | 92.66% | 84.58% | +8.1 |
| esci_us | 57.64% | 51.00% | +6.6 |
| sharc | 73.19% | 66.80% | +6.4 |
| clevr_count | 89.72% | 83.50% | +6.2 |
| aokvqa | 83.66% | 78.12% | +5.5 |
| vqav2_yesno | 85.88% | 80.82% | +5.1 |
| boolq | 87.71% | 82.97% | +4.7 |
| esci_es | 60.26% | 56.07% | +4.2 |
| contract_nli | 85.59% | 81.50% | +4.1 |
| xnli_en | 87.20% | 83.77% | +3.4 |
| esci_jp | 63.81% | 60.62% | +3.2 |
| xnli_zh | 78.20% | 75.58% | +2.6 |
| typed_decisions | 73.05% | 71.75% | +1.3 |
| massive_zh-CN | 76.82% | 75.56% | +1.3 |
| screenqa_noul | 72.35% | 71.14% | +1.2 |
| clevr_exist | 98.58% | 97.40% | +1.2 |
| banking77 | 73.70% | 72.95% | +0.7 |
| ag_news | 90.04% | 89.49% | +0.6 |
| clevr_attribute | 98.92% | 98.36% | +0.6 |
| emotion | 77.00% | 76.45% | +0.6 |
| wikiqa | 96.17% | 95.94% | +0.2 |
| mail_spam | 98.95% | 98.83% | +0.1 |
| sms_spam | 99.37% | 99.37% | −0.0 |
| mail_phishing | 98.95% | 99.24% | −0.3 |
| massive_en-US | 78.89% | 79.19% | −0.3 |
| screenqa_choice | 22.05% | 22.41% | −0.4 |

</details>

Vision and product-intent groups gain the most; nothing regresses beyond noise
(`screenqa_choice` is hard for both models at ~22%).

### Latency

Warm single-GPU BF16 predict latency, RTX 4090, merged LoRA (3 warmups, 20
repetitions; load/network excluded):

| Workload | p50 | Decisions/s |
| --- | ---: | ---: |
| Text, 1 question | 46.8 ms | 21.3 |
| Text, 50 questions | 122.4 ms | 407.7 |
| Image, 1 question | 49.9 ms | 19.9 |
| Image, 3 questions | 100.8 ms | 29.6 |

Upstream-published references on the same harness (RX 7900 XTX, not our
measurement — hardware differs): Dohnuts-0.1.0-0.8B text 1q 15.1 ms, text 50q
112.5 ms, image 1q 24.4 ms; Laya multilingual 9.8 ms text 1q, no image path;
Laya Vision 75.4 ms image 1q.

### JevBench v1.2.2 — 231 public tasks (measured)

Official runner, pinned datasets, rejections count as wrong:

| Model | Backbone | Overall | easy | standard | hard |
| --- | --- | ---: | ---: | ---: | ---: |
| **Linnaeus-0.1.0-2B** | Qwen3.5-2B | **73.16%** | 100% | 91.67% | 49.55% |
| open-alternative-jev | Qwen3.5-4B | 74.03% | 100% | 83.33% | 56.76% |
| system-one-open | Gemma 4 E2B LoRA | 73.16% | 100% | 93.06% | 48.65% |
| system-one | Qwen3-8B | 71.86% | 100% | 88.89% | 48.65% |
| Dohnuts-0.1.0-0.8B (upstream) | Qwen3.5-0.8B | 65.80% | — | — | — |
| Nimble 9B | 9B | 67.53% | 100% | 93.06% | 36.94% |
| jeff | GLiFormer 400M | 62.77% | 100% | 75.00% | 38.74% |
| Laya multilingual | ModernBERT-large 421M | 58.44% | 95.8% | 69.44% | 35.14% |
| openJev Verdict | ModernBERT-base 151M | 55.41% | 85.4% | 62.50% | 37.84% |
| GLiNER2 | gliner2.5-base | 58.01% | 97.9% | 63.89% | 36.94% |
| open-jev | DeBERTa-v3-large | 52.38% | 100% | 43.06% | 37.84% |

Reference points above us: SemIf (Qwen3.5-4B) 80.95%, Jev 1.13.0 86.58%,
GPT-5.6 Luna 97.40%. Among small local models we are at the top of the ~2B
class and ahead of every sub-1B classifier.

### Laya protocol suites (measured vs published)

Upstream-published baselines on the same frozen inputs; Linnaeus measured on
RTX 4090:

| Suite | Linnaeus-2B | Dohnuts-0.8B | Laya multilingual |
| --- | ---: | ---: | ---: |
| app.email_spam | 88.0% | 78.8% | **99.3%** |
| app.phishing | 87.5% | 75.8% | **99.3%** |
| app.guardrails_jailbreak | 83.8% | **90.5%** | 75.5% |
| app.moderation_toxicity | 64.0% | **68.2%** | 52.5% |
| app.rag_relevance | 67.5% | 66.5% | 65.7% |
| app.support_triage | 39.5% | 35.5% | **52.2%** |
| app.model_routing_domain | **86.2%** | 76.7% | 12.3% |
| jev.banking77_full | **73.2%** | 70.5% | 42.5% |
| jev.emotion | 77.5% | **78.2%** | 53.0% |
| colab/typed_decisions | **73.7%** | 72.3% | 34.2% |

### Multilingual suites (measured vs published)

| Suite | Linnaeus-2B | Dohnuts-0.8B | Laya multilingual |
| --- | ---: | ---: | ---: |
| MASSIVE intent (15 langs) | **77.4%** | 73.3% | 46.6% |
| MASSIVE scenario | **63.1%** | 52.6% | 44.7% |
| XNLI (15 langs) | **76.0%** | 70.9% | 73.8% |

Where upstream lost XNLI to Laya, Linnaeus-2B takes it back.

## Provenance

Linnaeus-0.1.0-2B was trained on a rented NVIDIA RTX 4090 with the upstream
recipe unchanged: rank-8 LoRA plus decision head, joint RLCD + cross-entropy,
3,600 steps, ~3.4 h of GPU training, checkpoint selected by development macro
accuracy.

## Acknowledgements

Linnaeus is a CUDA retarget of the upstream [Dohnuts](https://github.com/PsiACE/dohnuts)
project by PsiACE. The training recipe (RLCD + cross-entropy), data mixture,
isolation rules, evaluation protocol, and execution design are theirs; the
baseline numbers quoted throughout come from their published artifacts.

## License

The code is licensed under [Apache-2.0](LICENSE). Model weights are released
under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for
non-commercial research. See the [model card](MODEL_CARD.md#license) for
checkpoint terms and training-data restrictions.
