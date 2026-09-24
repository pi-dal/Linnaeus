"""Collect one completed run into a checkpoint card, report, and plotting tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from dohnuts.metrics import by_dataset


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
            )
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/v1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run / "checkpoint"
    display_names = {
        "dohnuts": "Dohnuts",
        "laya": "Laya multilingual",
        "laya-vision": "Laya Vision",
    }
    metadata = read_json(checkpoint / "dohnuts.json")
    selected = Path(metadata["selected_run"])
    evaluation = read_json(selected / "evaluation.json")
    tables = {
        name: []
        for name in [
            "training",
            "development",
            "quality",
            "reliability",
            "primitive-candidate-slices",
            "latency-samples",
        ]
    }
    for row in read_rows(selected / "metrics.jsonl"):
        if row["kind"] == "train":
            tables["training"].append(
                {
                    "run": selected.name,
                    **{k: v for k, v in row.items() if isinstance(v, (int, float, str))},
                    **{"memory_" + k: v for k, v in row["memory"].items()},
                }
            )
        elif row["kind"] == "dev":
            for dataset, metrics in row["metrics"].items():
                if isinstance(metrics, dict):
                    tables["development"].append(
                        {
                            "run": selected.name,
                            "step": row["step"],
                            "dataset": dataset,
                            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                        }
                    )
    for mode in ["uncalibrated", "calibrated"]:
        for dataset, metrics in evaluation[mode].items():
            if not isinstance(metrics, dict):
                continue
            info = {
                "run": selected.name,
                "dataset": dataset,
                "calibration": mode,
                "selected_step": evaluation["selected_step"],
            }
            tables["quality"].append(
                {
                    **info,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            )
            tables["reliability"].extend({**info, **bucket} for bucket in metrics["reliability"])
    tables["primitive-candidate-slices"] = [
        {"run": selected.name, **{k: v for k, v in row.items() if k != "reliability"}}
        for row in evaluation["primitive_candidate_slices"]
    ]
    latency = []
    for path in sorted((args.run / "benchmarks").glob("*.jsonl")):
        for row in read_rows(path):
            if row["kind"] != "matched_inference":
                continue
            row = {**row, "engine": path.stem}
            latency.append(row)
            for repeat, duration in enumerate(row["end_to_end"]["samples_ms"]):
                tables["latency-samples"].append(
                    {
                        "engine": row["engine"],
                        "case": row["case"],
                        "questions": row["questions"],
                        "repeat": repeat,
                        "end_to_end_ms": duration,
                        "input_shape": json.dumps(row["input_shape"]),
                        **row["memory"],
                        **{
                            f"telemetry_{k}_{stat}": value
                            for k, values in row["telemetry"].items()
                            for stat, value in values.items()
                        },
                    }
                )
    resources = selected / "resources.jsonl"
    if resources.exists():
        tables["training-resources"] = read_rows(resources)
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)

    predictions = read_rows(selected / "test-predictions.jsonl")
    own_ids = {row["id"] for row in predictions}
    references = {}
    for engine in ["laya", "laya-vision"]:
        directory = args.run / "references" / engine
        if not (directory / "predictions.jsonl").exists():
            continue
        ids = {row["id"] for row in read_rows(directory / "predictions.jsonl")}
        if not ids <= own_ids:
            raise ValueError(f"{engine} includes IDs absent from the Dohnuts final predictions")
        references[engine] = {
            "dohnuts_on_same_ids": by_dataset(
                [r for r in predictions if r["id"] in ids], metadata["temperatures"]
            ),
            **read_json(directory / "report.json"),
        }
    summary = {
        "checkpoint": metadata,
        "evaluation": evaluation,
        "same_id_references": references,
        "latency": latency,
    }
    # Optional stages may have been skipped via --skip; include what exists.
    for key, path in {
        "acceptance": args.run / "acceptance/report.json",
        "quality_audit": args.run / "quality-audit/report.json",
        "jevbench": args.run / "jevbench/summary.json",
        "laya_chart": args.run / "laya-chart/summary.json",
    }.items():
        if path.exists():
            summary[key] = read_json(path)
    summary |= {
        "scope": [
            "One seed; variation across seeds is unmeasured. Development selects weights; calibration fits temperatures; test never selects either.",
            "Laya references use their own templates, FP32 CPU weights and published temperatures; Dohnuts uses merged BF16 weights.",
            "Laya Vision has VQAv2 source-pool and A-OKVQA selection exposure. Backbone pretraining exposure is unverified.",
            "Bub acceptance verifies local decision-tool calls, not autonomous planning quality.",
            "Source model, dataset and image terms apply; this report does not assign a new weight license.",
        ],
    }
    comparisons, changes = [], []
    for baseline in sorted((args.run / "baselines").glob("step-*")):
        prior = read_json(baseline / selected.name / "evaluation.json")
        if prior["source_hashes"] != evaluation["source_hashes"]:
            raise ValueError("Training-budget comparison requires identical dataset partitions")
        comparison = {
            "baseline": str(baseline),
            "before_step": prior["selected_step"],
            "after_step": evaluation["selected_step"],
            "before_accuracy": prior["calibrated"]["macro_accuracy"],
            "after_accuracy": evaluation["calibrated"]["macro_accuracy"],
        }
        prior_jev_path = baseline / "jevbench/summary.json"
        if prior_jev_path.exists() and "jevbench" in summary:
            comparison["before_jev_correct"] = read_json(prior_jev_path)["metrics"]["n_correct"]
            comparison["after_jev_correct"] = summary["jevbench"]["metrics"]["n_correct"]
        comparisons.append(comparison)
        for name, value in evaluation["calibrated"].items():
            if not isinstance(value, dict):
                continue
            before = prior["calibrated"][name]
            changes.append(
                {
                    "baseline": str(baseline),
                    "dataset": name,
                    "n": value["n"],
                    **{f"before_{key}": before[key] for key in ["accuracy", "nll", "ece_15"]},
                    **{f"after_{key}": value[key] for key in ["accuracy", "nll", "ece_15"]},
                    "accuracy_delta_pp": 100 * (value["accuracy"] - before["accuracy"]),
                }
            )
    summary["training_budget_comparison"] = comparisons
    write_csv(output / "training-budget-comparison.csv", changes)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# {metadata['model_id']}",
        "",
        (
            f"Built on {metadata['base_model']}, this checkpoint returns decision distributions from text "
            "and images. It supports candidate selection (`choice`), truth estimates (`noul`), "
            "and ordered scores (`score`) without generating reasoning or free-form responses."
        ),
        "",
        "## Model details",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Base model | {metadata['base_model']} |",
        f"| Selection | Seed {metadata['seed']}, update {metadata['selected_step']:,} |",
        "| Inference | " + metadata["inference"] + " |",
        "| Inputs | Text and one PIL image; 2–128 candidates; 4,096 tokens per question |",
        "",
        (
            "Questions reuse a shared input prefix and compute their suffixes in parallel. "
            "Additional questions still require computation."
        ),
        "",
        "## Training",
        "",
        (
            "The training recipe combines RLCD and auxiliary cross-entropy, following the "
            "pinned Laya and Laya Vision references. Temperature calibration uses an independent "
            "partition after LoRA merging. Calibration quality is measured below."
        ),
        "",
        (
            "[Training recipe](../recipe.json) · "
            "[RLCD implementation and upstream attribution](../../../docs/rlcd.md)"
        ),
        "",
        "## Evaluation",
        "",
        f"Held-out macro accuracy: {evaluation['calibrated']['macro_accuracy']:.2%}.",
        "",
    ]
    figure_manifest = args.run / "figures/manifest.json"
    if (
        figure_manifest.exists()
        and read_json(figure_manifest)["checkpoint_sha256"] == metadata["weights_sha256"]
    ):
        lines += [
            (
                "[Comparison figures: PNG, SVG and PDF](../figures/README.md). "
                "Each figure states its dataset scope, reference checkpoint and measurement conditions."
            ),
            "",
            "![JevBench public-task comparison](../figures/01_jevbench.png)",
            "",
        ]
    lines += [
        "| Dataset | N | Accuracy | NLL before / after calibration | ECE before / after |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, value in evaluation["calibrated"].items():
        if isinstance(value, dict):
            raw = evaluation["uncalibrated"][name]
            lines.append(
                f"| {name} | {value['n']} | {value['accuracy']:.2%} | "
                f"{raw['nll']:.4f} / {value['nll']:.4f} | "
                f"{raw['ece_15']:.4f} / {value['ece_15']:.4f} |"
            )
    if references:
        lines += [
            "",
            "### Reference comparisons",
            "",
            "| Same-ID reference | Dohnuts macro accuracy | Reference macro accuracy |",
            "| --- | ---: | ---: |",
        ]
    for name, value in references.items():
        lines.append(
            f"| {display_names.get(name, name)} | {value['dohnuts_on_same_ids']['macro_accuracy']:.2%} | "
            f"{value['metrics']['macro_accuracy']:.2%} |"
        )
    lines += [
        "",
        "### Inference speed",
        "",
        "Warm RX 7900 XTX end-to-end predict latency, including preprocessing and transfers.",
        "Three warmups and 20 synchronized repetitions; network and queueing excluded.",
        "",
        "| Engine | Workload | p50 ms | p95 ms | Decisions/s |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in latency:
        timing = row["end_to_end"]
        lines.append(
            f"| {display_names.get(row['engine'], row['engine'])} | {row['case']} | {timing['p50_ms']:.2f} | "
            f"{timing['p95_ms']:.2f} | {timing['decisions_per_second']:.1f} |"
        )
    lines += ["", "### Benchmark coverage", ""]
    if "jevbench" in summary:
        jev = summary["jevbench"]
        lines.append(
            f"JevBench: {jev['metrics']['n_correct']}/{jev['coverage']['available']} public tasks correct; "
            f"{jev['coverage']['unavailable']} official tasks unavailable. No official leaderboard rank."
        )
    if "laya_chart" in summary:
        laya = summary["laya_chart"]
        lines.append(
            f"Laya chart protocol: {len(laya['metrics'])} suites, including 51 MASSIVE languages; "
            f"MASSIVE macro accuracy {laya['massive51']['macro_accuracy']:.2%}."
        )
    lines += [
        "",
        "## Use",
        "",
        "```python",
        "from dohnuts.predictor import Predictor",
        f'model = Predictor.from_checkpoint("{checkpoint}")',
        "```",
        "",
        "[Installation, question definitions and response fields](../../../docs/inference.md).",
        "",
        "## Limitations",
        "",
        (
            "For `choice` and `score`, `confidence` is `1 - H(p) / log(K)`. For `noul`, "
            "it is `max(p, 1 - p)`. These distribution summaries are not empirical correctness "
            "guarantees; calibration metrics use maximum probability and observed correctness."
        ),
        "Raw predictions, resource samples, calibration bins and quality diagnostics accompany this report.",
        "",
        *[f"- {note}" for note in summary["scope"]],
        "",
        "## Reproducibility",
        "",
        f"Base revision: `{metadata['base_revision']}`.",
        f"Weight SHA-256: `{metadata['weights_sha256']}`.",
        "Recipe SHA-256: `"
        + hashlib.sha256((args.run / "recipe.json").read_bytes()).hexdigest()
        + "`.",
        (
            "[Recorded source-file hashes](../source-sha256.json) identify the code snapshot "
            "used for the run. A training Git revision is not recorded."
        ),
    ]
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report)
    (checkpoint / "MODEL_CARD.md").write_text(report)
    print(json.dumps({"report": str(output / "report.md"), "checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    main()
