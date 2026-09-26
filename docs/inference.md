# Installation and inference

The Linnaeus runtime uses Python 3.12 and PyTorch 2.9.1 with CUDA 12.8 on an
NVIDIA GPU (Ampere or newer for BF16). Upstream measured ROCm 6.4 on an
AMD Radeon RX 7900 XTX. Install [mise](https://mise.jdx.dev/) (it provisions
Python 3.12 and uv from `mise.toml`), then install from source:

```bash
git clone <linnaeus-remote>
cd Linnaeus
mise install
uv sync
```

The lock file targets Python 3.12 on Linux (`required-environments`). The
`tool.uv.sources` entries in `pyproject.toml` bind PyTorch and torchvision to
the explicit CUDA 12.8 index; Triton resolves from PyPI.
See [development](development.md) for dependency groups, checks, and updating
the lock.

## Load a checkpoint

Load the model by its Hugging Face repository ID. The loader downloads the
decision weights and the exact Qwen3.5-0.8B revision recorded in the checkpoint.
Downloads use the Hugging Face cache and are reused by later calls. Both
repositories are public; downloading them does not require a Hugging Face login.

Run Python examples with `uv run python` from the repository root:

```python
from linnaeus.predictor import Predictor

model = Predictor.from_checkpoint("pi-dal/Linnaeus-0.1.0-2B")
```

The loader verifies the checkpoint's weight checksum and base revision, merges
LoRA, and applies its saved calibration temperatures. The checkpoint contains
the decision head as well as LoRA; load it through Linnaeus. It is not a standalone
Transformers language model or a standard PEFT adapter export.

The same method accepts a local checkpoint directory. An optional `revision`
pins a Hub commit, and `base_model` supplies an existing local base with its
recorded `revision.txt`. `model.metadata` exposes the loaded model identity,
weight checksum, and calibration. Weights are not bundled with the Python package.

## Ask questions

Pass a shared state and a mapping of question names to question definitions:

```python
result = model.predict(
    {"message": "The replacement arrived, but it is broken too."},
    {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"support": "Product issues", "billing": "Payment issues"},
        },
        "repeat_issue": {
            "type": "noul",
            "instructions": "Has the customer experienced this problem before?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently does this request need attention?",
            "criteria": ["routine", "soon", "immediate"],
        },
    },
)

answers = result["answers"]
```

Questions are independent: each sees the state and its own instructions and
candidates. A question cannot refer to another question's answer in the same call.

| Type | Criteria | Answer fields |
| --- | --- | --- |
| `choice` | Candidate list, or label-to-description mapping | `choice` and `probabilities`, keyed by candidate label |
| `noul` | Optional descriptions under `false` and `true` | `noul`, the estimated probability that the statement is true |
| `score` | Level descriptions in ascending order | `score`, the expected zero-based level, and `probabilities`, keyed by level index |

Each answer also includes `type` and `confidence`. For `choice` and `score`,
confidence is normalized inverse entropy, `1 - H(p) / log(K)`; for `noul`, it is
`max(p, 1 - p)`. This describes how concentrated the distribution is. It is not
an empirical probability that the decision is correct.

The response includes `model` and `usage` alongside `answers`. Usage records
input tokens and image count.

## Include an image

Supply one decoded PIL image under the state's `image` key:

```python
from PIL import Image

with Image.open("parcel.jpg") as image:
    result = model.predict(
        {"image": image.convert("RGB")},
        {"damage": {"type": "noul", "instructions": "Is the packaging damaged?"}},
    )
```

Each question supports 2–128 candidates and a 4,096-token input budget,
including state, instructions, candidates, and image tokens. Inputs that exceed
the budget are rejected; shorten the state or candidate descriptions and retry.
Image features and the shared input prefix are reused where possible. More
questions still require more computation.

## Apple Silicon (MLX / on-device)

The published MLX builds run the model natively on macOS and iOS via
[mlx-lm](https://github.com/ml-explore/mlx-lm) /
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).
The `-MLX-*` builds are text-only; the `-MLX-VLM-*` builds (converted via
mlx-vlm) also keep the vision tower for `state['image']` decisions:

| Artifact | Size | JevBench v1.2.2 (231 tasks) | Use |
| --- | --- | --- | --- |
| `pi-dal/Linnaeus-0.1.0-2B-merged` | 4.3 GB | 71.0% (torch MPS) | Mac dev, conversion source |
| `pi-dal/Linnaeus-0.1.0-2B-MLX-8bit` | 1.9 GB | 70.56% | Mac / iPhone, text quality pick |
| `pi-dal/Linnaeus-0.1.0-2B-MLX-4bit` | 1.0 GB | 67.53% | iPhone, text size pick |
| `pi-dal/Linnaeus-0.1.0-2B-MLX-VLM-8bit` | 2.5 GB | 70.56% text + **images** | Mac / iPhone multimodal |
| `pi-dal/Linnaeus-0.1.0-2B-MLX-VLM-4bit` | 1.6 GB | ~67% text + **images** | iPhone multimodal, size pick |

The VLM builds are converted with `mlx-vlm convert` (vision tower stays
bf16; only the language stack is quantized). A post-conversion step must
write **both** `quantization` and `quantization_config` into `config.json`
({"group_size": 64, "bits": N, "mode": "affine"}) — mlx-vlm 0.7.1 omits
them, `load_weights` then rejects the quantized tensors, and Hugging Face
misclassifies the repo as a finetune instead of a quantization. Pass a
decoded PIL image via `state['image']` to the same `MlxPredictor.predict()`
contract.

CUDA reference on the same tasks is 73.16%; the residual gap is the fla
chunked delta-rule kernel vs. reference implementations, concentrated on
hard-tier borderline tasks (easy/standard are identical). Upstream's
published result on the same benchmark is 65.80%.

```python
from linnaeus.mlx_predictor import MlxPredictor

predictor = MlxPredictor("runs/2b/exports/linnaeus-2b-8bit")
result = predictor.predict(state, questions)  # same contract as Predictor
```

To reproduce the exports:

```bash
python scripts/export_merged_hf.py --base <Qwen3.5-2B snapshot> \
    --adapter runs/2b/checkpoint --out runs/2b/exports/merged-hf
# text-only build
uvx --from mlx-lm python -m mlx_lm convert \
    --hf-path runs/2b/exports/merged-hf \
    --mlx-path runs/2b/exports/linnaeus-2b-8bit -q --q-bits 8
# multimodal build (keeps vision tower)
python -m mlx_vlm convert --hf-path runs/2b/exports/merged-hf \
    --mlx-path runs/2b/exports/linnaeus-2b-vlm-8bit --quantize --q-bits 8
```

The merged checkpoint embeds the scalar decision head as an extra
`embed_tokens`/`lm_head` row (`score_row_id`), so a stock LM forward is all a
runtime needs: read `logits[marker_pos, score_row_id]` at each
`<|fim_suffix|>` marker, apply the per-type temperature, softmax across a
question's candidates. `linnaeus-runtime.json` carries that contract to
non-Python runtimes (e.g. a ~100-line Swift shim for iOS).
