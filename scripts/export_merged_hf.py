"""Merge the trained LoRA adapter into the base model and emit a standard HF checkpoint.

Two artifacts in one pass:
1. LoRA merge: W += (lora_B @ lora_A) * (alpha / r) for every adapter pair.
2. Vocab extension: the scalar decision head (1 x hidden) is appended as an
   extra row to `embed_tokens`. Since the base model ties embeddings, the same
   row becomes the lm_head row — so `logits[..., vocab_size]` at a marker
   position equals the raw decision score. Any stock LM runtime (MLX,
   llama.cpp, vLLM) can then score candidates with zero custom head code.

The extra token is never fed as input; it exists only so its lm_head row
carries the decision head. `vocab_size` in text_config is bumped accordingly.

Usage:
    python scripts/export_merged_hf.py \
        --base ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/<rev> \
        --adapter runs/2b/checkpoint --out runs/2b/exports/merged-hf
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="Base HF snapshot dir")
    ap.add_argument("--adapter", type=Path, required=True, help="Checkpoint dir (linnaeus.json + adapter.safetensors)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads((args.adapter / "linnaeus.json").read_text())
    rank = meta["lora_rank"]
    scaling = (rank * 2) / rank  # lora_alpha = rank*2, lora_dropout = 0
    adapter = load_file(args.adapter / "adapter.safetensors")
    head = adapter.pop("head.weight").to(torch.bfloat16)  # [1, hidden]
    assert head.shape[0] == 1

    # Regroup LoRA pairs: leaf module -> {A, B}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    pat = re.compile(
        r"backbone\.language_model\.base_model\.model\.(.+)\.lora_([AB])\.default\.weight"
    )
    for key in adapter:
        m = pat.fullmatch(key)
        assert m, f"unexpected adapter tensor name: {key}"
        pairs.setdefault(m.group(1), {})[m.group(2)] = adapter[key]
    print(f"{len(pairs)} LoRA pairs to merge (scale={scaling})")

    # Load the full base state dict (single shard on disk).
    index = json.loads((args.base / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(args.base / shard))

    merged: dict[str, torch.Tensor] = {}
    used = set()
    for name, w in state.items():
        leaf = name.removeprefix("model.language_model.").removesuffix(".weight")
        if name.startswith("model.language_model.layers.") and leaf in pairs:
            p = pairs[leaf]
            delta = (p["B"].to(torch.float32) @ p["A"].to(torch.float32)) * scaling
            merged[name] = (w.to(torch.float32) + delta).to(torch.bfloat16)
            used.add(leaf)
        else:
            merged[name] = w
    assert used == set(pairs), f"unapplied LoRA pairs: {set(pairs) - used}"

    # Vocab extension: append the decision head as an extra embedding row.
    emb_name = "model.language_model.embed_tokens.weight"
    emb = merged[emb_name]
    assert head.shape[1] == emb.shape[1], (head.shape, emb.shape)
    merged[emb_name] = torch.cat([emb, head], dim=0).contiguous()
    print(f"embed_tokens {emb.shape} -> {merged[emb_name].shape} (extra row = decision head)")

    # Write merged checkpoint.
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(merged, args.out / "model.safetensors", metadata={"format": "pt"})
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = args.base / name
        if src.exists():
            shutil.copy(src, args.out / name)

    config = json.loads((args.out / "config.json").read_text())
    config["text_config"]["vocab_size"] = emb.shape[0] + 1
    config["linnaeus_score_row"] = emb.shape[0]  # runtime contract hint
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    contract = {
        "model_id": meta["model_id"],
        "adapter_sha256": meta["weights_sha256"],
        "score_row_id": emb.shape[0],
        "marker": "<|fim_suffix|>",
        "temperatures": meta["temperatures"],
        "question_types": meta.get("question_types"),
        "max_length": meta["max_length"],
    }
    (args.out / "linnaeus-runtime.json").write_text(json.dumps(contract, indent=2))
    print(f"merged checkpoint -> {args.out}")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
