# Deploy on OpenBayes (NVIDIA)

Field notes from the Linnaeus-0.1.0-2B run on an OpenBayes RTX 4090 workspace.
Every trap below cost real GPU hours.

## Storage model

| Path | Type | Survives restart | Notes |
| --- | --- | --- | --- |
| `/` (overlay) | local NVMe, ~100 GB | **no** | fast; use for data + caches |
| `/openbayes/home` → `/output` | CephFS | yes | counts against the 50 GB account quota |
| `/openbayes/input/inputN` | dataset mount | yes | `-d` CLI bindings are **read-only**; rw is console-only |
| `HF_HOME=/output/huggingface` | CephFS | yes | image default; HF downloads persist across restarts |

Recommended layout: `data/raw`, `data/processed`, `.cache` are symlinks to
`/root/dh/*` on scratch; repo + `.venv` + `runs/` live on `/openbayes/home`.
Peak scratch usage during `prepare-data` was ~68 GB.

## China-network mirrors

`huggingface.co` is unreachable; several CDNs are throttled. Working matrix:

| Source | Works | Speed | Fallback |
| --- | --- | --- | --- |
| hf-mirror datasets resolve | yes | ~18 MB/s | — |
| hf-mirror model resolve | flaky | 27 MB/s or stalls | **ModelScope** `modelscope.cn/models/{org}/{name}/resolve/master/{file}` ~10–44 MB/s |
| xet CAS (`cas-server.xethub.hf.co`) | **no** | — | `HF_HUB_DISABLE_XET=1` |
| storage.googleapis.com, media.githubusercontent.com | throttled | ~18 KB/s | download via local proxy, rsync up |
| github.com | **no** | — | clone locally, tar over ssh |
| pypi | yes | ok | `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` |

`HF_ENDPOINT` only affects the huggingface_hub SDK. `prepare_data.py` downloads
manifest URLs with curl — it rewrites `huggingface.co` hosts via
`HF_ENDPOINT` itself (`download_url`); sha256 provenance is unchanged.

`download_file` passes `--speed-limit 10240 --speed-time 30` to curl so a
trickling-but-alive connection aborts and retries instead of hanging for hours.

## Environment

```bash
curl -fsSL https://mise.run | sh            # ~/.local/bin is scratch; MISE_DATA_DIR persists tools
export MISE_DATA_DIR=/openbayes/home/.mise
export PATH="$HOME/.local/bin:$MISE_DATA_DIR/shims:$PATH"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export LINNAEUS_VRAM_FRACTION=0.95
uv sync --extra train --extra agent --extra telemetry --group quality
```

The plain `uv sync` has no extras — `sklearn`, agent extras, NVML telemetry
all come from the extras list above. If `mise` shims lose their version pin
after a restart, call the tool directly:
`.mise/installs/uv/0.11/.mise-bins/uv`.

## Stage resume semantics (read before restarting)

`completed.json` skips a stage only when `done[name] == command` **exactly**.
Consequences:

- Changing the package name (`dohnuts` → `linnaeus`) invalidated every recorded
  `-m dohnuts.train` command → `evaluate-42` re-ran its full 180k-row eval.
  After any rename, either migrate completed.json or accept re-runs.
- `train-*` resumes from `last.pt` (saved every `save_every` steps); a
  "re-run" at the final step is a no-op.
- `run_laya_benchmark` resumes per-case from `predictions.jsonl`.
- `prepare-data` is forced to re-check whenever processed splits are missing;
  verified manifest files are hash-skipped, so re-runs only fetch what is gone.
- Evaluation is **dataloader-bound**, not GPU-bound (0–30% GPU util on image
  groups). `--workers 16` cut the 26-group dev eval from ~45 min to ~11 min.

## bayes CLI operations

- `bayes gear run workspace -d "<user>/jobs/<jobId>/output:/output"` mounts a
  previous job's persistent output into a new container — the way to keep
  working after quota loss on the original resource or to fetch files from a
  cheap `cpu` container.
- `bayes data upload <dataset> -v <ver> -p <path>` works reliably from a local
  machine; the same CLI hung mid-upload from inside a container. For many
  small files, tar first.
- SSH port changes on every restart → `ssh-keygen -R "[host]:port"` then
  `StrictHostKeyChecking=accept-new`.
- Workspace jobs were observed being cancelled after ~13 h of runtime; keep
  valuable output on persistent paths continuously, not at the end.
- GPU free quota is per-resource and depletes; all single-GPU types can show
  无算力额度 simultaneously while `cpu` still works.
- rsync over the job SSH link can drop mid-stream on larger trees; a
  `tar czf | ssh 'tar xzf -'` pipe was more reliable.
