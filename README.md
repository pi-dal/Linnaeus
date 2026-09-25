<img src="assets/linnaeus-logo.png" width="160" align="right" alt="A pink-frosted doughnut doing a facepalm">

# Linnaeus

**NVIDIA-optimized fork of [Dohnuts](https://github.com/PsiACE/dohnuts).**

Upstream trains and evaluates on AMD ROCm (RX 7900 XTX). Linnaeus retargets the
same recipe to CUDA servers:

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

Upstream reference: [Dohnuts-0.1.0-0.8B](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B).

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
The [model card](MODEL_CARD.md) reports quality, latency, and limitations, with
comparisons against Jev, Laya multilingual, and Laya Vision. For agents,
[Bub integration](docs/bub-agent.md) exposes the same interface as one decision
tool through the Bub SDK.

![Dohnuts 0.1.0 model overview and benchmarks](docs/figures/overview.svg)

## Built at home

We developed, trained, calibrated, and evaluated Dohnuts on a home PC with one
AMD Radeon RX 7900 XTX (24 GB). We used hardware we already owned and a $0
additional project budget. The main ingredient was time.

## License

The code is licensed under [Apache-2.0](LICENSE). The model weights are licensed
under [CC BY-NC-SA 4.0](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B/blob/main/LICENSE)
for non-commercial research. See the [model card](MODEL_CARD.md#license) for
checkpoint terms and training-data restrictions.
