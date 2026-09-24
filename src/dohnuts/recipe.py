"""The Dohnuts training and evaluation recipe.

Values are recorded in each run for reproducibility. They are product defaults,
not a configuration surface. Model-specific behavior belongs in an adapter.
"""

from pathlib import Path

from dohnuts.rlcd import RLCDConfig

IMAGE_PIXELS = 512**2
MAX_LENGTH = 2048
TRAINING_STEPS = 3600
LR_DECAY_STEPS = 2400
BASE_MODEL = Path(".cache/models/Qwen3.5-0.8B")
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DATA = Path("data/processed/v1")


def training_recipe(
    *,
    model=BASE_MODEL,
    data=DATA,
    seed=42,
    rlcd=None,
    steps=TRAINING_STEPS,
    base_model_id=BASE_MODEL_ID,
    lora_rank=8,
    workers=2,
    eval_batch_size=16,
    cpu_threads=8,
):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("Training steps must be a positive integer")
    policy = rlcd or RLCDConfig()
    recipe = {
        "model": str(model),
        "base_model_id": base_model_id,
        "data": str(data),
        "seed": seed,
        "lora_rank": lora_rank,
        "batch_size": 8,
        "accumulation": 4,
        "steps": steps,
        "backbone_lr": 1e-4,
        "head_lr": 5e-4,
        "train_cap": 6000,
        "dev_cap": 256,
        "image_pixels": IMAGE_PIXELS,
        "max_length": MAX_LENGTH,
        "cpu_threads": cpu_threads,
        "workers": workers,
        "eval_batch_size": eval_batch_size,
        "log_every": 10,
        "eval_every": 400,
        "save_every": 100,
        "backend": {
            "linear_patch": True,
            "triton_convolution": True,
            "experimental_rocm_sdpa": True,
            "fused_norm_and_swiglu": True,
            "shared_prefix": True,
            "frozen_vision_cache_MiB": 128,
        },
    }
    if policy != RLCDConfig():
        recipe["rlcd"] = {"sigma": policy.sigma, "ce_weight": policy.ce_weight}
    return recipe
