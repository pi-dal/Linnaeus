# Linnaeus-0.1.0-2B

Built on Qwen/Qwen3.5-2B, this checkpoint returns decision distributions from text and images. It supports candidate selection (`choice`), truth estimates (`noul`), and ordered scores (`score`) without generating reasoning or free-form responses.

## Model details

| Property | Value |
| --- | --- |
| Base model | Qwen/Qwen3.5-2B |
| Selection | Seed 42, update 2,800 |
| Inference | merged LoRA, BF16, fused operations, shared-prefix parallel candidate scoring |
| Inputs | Text and one PIL image; 2–128 candidates; 4,096 tokens per question |

Questions reuse a shared input prefix and compute their suffixes in parallel. Additional questions still require computation.

## Training

The training recipe combines RLCD and auxiliary cross-entropy, following the pinned Laya and Laya Vision references. Temperature calibration uses an independent partition after LoRA merging. Calibration quality is measured below.

[Training recipe](../recipe.json) · [RLCD implementation and upstream attribution](../../../docs/rlcd.md)

## Evaluation

Held-out macro accuracy: 80.78%.

| Dataset | N | Accuracy | NLL before / after calibration | ECE before / after |
| --- | ---: | ---: | ---: | ---: |
| ag_news | 7600 | 90.04% | 0.8728 / 0.5378 | 0.0832 / 0.0710 |
| aokvqa | 1138 | 83.66% | 0.6347 / 0.4746 | 0.0930 / 0.0513 |
| banking77 | 3080 | 73.70% | 1.0536 / 1.1711 | 0.0275 / 0.2001 |
| boolq | 3270 | 87.71% | 0.3854 / 0.3288 | 0.0610 / 0.0648 |
| clevr_attribute | 53734 | 98.92% | 0.1687 / 0.1230 | 0.0100 / 0.0095 |
| clevr_count | 35422 | 89.72% | 0.7159 / 0.2652 | 0.0793 / 0.0077 |
| clevr_exist | 20196 | 98.58% | 0.2526 / 0.0955 | 0.0137 / 0.0127 |
| contract_nli | 1173 | 85.59% | 0.5014 / 0.4020 | 0.0888 / 0.0338 |
| emotion | 2000 | 77.00% | 0.7237 / 0.6812 | 0.0777 / 0.0488 |
| esci_es | 1482 | 60.26% | 0.9241 / 0.9775 | 0.0342 / 0.1266 |
| esci_jp | 1633 | 63.81% | 0.8878 / 0.9531 | 0.0349 / 0.1410 |
| esci_us | 1145 | 57.64% | 0.9185 / 0.9712 | 0.0489 / 0.0868 |
| mail_phishing | 1050 | 98.95% | 0.1327 / 0.0467 | 0.0097 / 0.0094 |
| mail_spam | 854 | 98.95% | 0.1658 / 0.0580 | 0.0105 / 0.0098 |
| massive_en-US | 2970 | 78.89% | 0.7713 / 0.7944 | 0.0532 / 0.1141 |
| massive_zh-CN | 2921 | 76.82% | 0.8801 / 0.8552 | 0.0664 / 0.0904 |
| scienceqa | 2017 | 92.66% | 0.2942 / 0.2049 | 0.0489 / 0.0342 |
| screenqa_choice | 848 | 22.05% | 2.3429 / 2.4277 | 0.0464 / 0.0603 |
| screenqa_noul | 2148 | 72.35% | 0.5747 / 0.6116 | 0.0341 / 0.1288 |
| sharc | 8276 | 73.19% | 0.6671 / 0.6700 | 0.0705 / 0.0547 |
| sms_spam | 794 | 99.37% | 0.0910 / 0.0319 | 0.0062 / 0.0061 |
| typed_decisions | 2000 | 73.05% | 0.9068 / 0.9944 | 0.1073 / 0.2750 |
| vqav2_yesno | 8102 | 85.88% | 0.3962 / 0.4835 | 0.0248 / 0.1920 |
| wikiqa | 6160 | 96.17% | 0.4844 / 0.1710 | 0.0374 / 0.0322 |
| xnli_en | 5009 | 87.20% | 0.3602 / 0.3727 | 0.0307 / 0.0601 |
| xnli_zh | 5009 | 78.20% | 0.5986 / 0.5582 | 0.0824 / 0.0297 |

### Inference speed

Warm RTX 4090 end-to-end predict latency, including preprocessing and transfers.
Three warmups and 20 synchronized repetitions; network and queueing excluded.

| Engine | Workload | p50 ms | p95 ms | Decisions/s |
| --- | --- | ---: | ---: | ---: |
| Linnaeus | vision_protocol_text_1q | 44.53 | 45.29 | 22.4 |
| Linnaeus | vision_protocol_text_3q | 47.81 | 48.29 | 62.9 |
| Linnaeus | vision_protocol_image_1q | 49.88 | 51.27 | 19.9 |
| Linnaeus | vision_protocol_image_3q | 100.83 | 102.87 | 29.6 |
| Linnaeus | distinct_text_1q | 46.83 | 48.00 | 21.3 |
| Linnaeus | distinct_text_5q | 94.32 | 95.89 | 52.8 |
| Linnaeus | distinct_text_10q | 94.83 | 96.16 | 105.1 |
| Linnaeus | distinct_text_50q | 122.42 | 124.17 | 407.7 |

### Benchmark coverage


## Use

```python
from linnaeus.predictor import Predictor

model = Predictor.from_checkpoint("runs/2b/checkpoint")
```

[Installation, question definitions and response fields](../../../docs/inference.md).

## Limitations

For `choice` and `score`, `confidence` is `1 - H(p) / log(K)`. For `noul`, it is `max(p, 1 - p)`. These distribution summaries are not empirical correctness guarantees; calibration metrics use maximum probability and observed correctness.
Raw predictions, resource samples, calibration bins and quality diagnostics accompany this report.

- One seed; variation across seeds is unmeasured. Development selects weights; calibration fits temperatures; test never selects either.
- Laya references use their own templates, FP32 CPU weights and published temperatures; Linnaeus uses merged BF16 weights.
- Laya Vision has VQAv2 source-pool and A-OKVQA selection exposure. Backbone pretraining exposure is unverified.
- Bub acceptance verifies local decision-tool calls, not autonomous planning quality.
- Source model, dataset and image terms apply; this report does not assign a new weight license.

## Reproducibility

Base revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
Weight SHA-256: `0424555feea3126ee02d8d1b12636fad8cf491a5964a00a04b2fec67bcc0f602`.
Recipe SHA-256: `d0017a70f1b6c789318700339cd454a74a05a3cac9f0687c1c322bd6eb031c62`.
[Recorded source-file hashes](../source-sha256.json) identify the code snapshot used for the run. A training Git revision is not recorded.
