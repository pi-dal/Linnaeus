# Development

[mise](https://mise.jdx.dev/) provisions the toolchain (`mise.toml`: Python
3.12 + uv) and runs every project command (`mise run <task>`). The runtime
targets the Linux/CUDA environment documented in the
[inference guide](inference.md); the lock is restricted to
`sys_platform == 'linux'` via `required-environments`. The quality and
documentation groups can be installed without model dependencies on a Linux
host.

```bash
mise install
mise run install
mise run check
mise run test
mise run build
```

`mise run install` installs the runtime, training, agent and telemetry extras,
and quality tools. The site lives in `site/` (Nuxt + UnoCSS): `mise run site-preview`.
Run `mise tasks` to see the available commands.

## Dependencies

`pyproject.toml` owns the package metadata, dependencies, tool settings, and uv
configuration. `uv.lock` records the resolution for Python 3.12 on Linux,
including the pinned Bub Git revision and the `+cu128` PyTorch artifacts bound
through `tool.uv.sources` and the explicit `pytorch-cu128` index. The published
experiment's environment snapshot remains historical evidence, not an
installation input.

| Selection | Purpose |
| --- | --- |
| Default dependencies | Model runtime |
| `train` extra | Training, evaluation, and the unit tests' data dependencies |
| `telemetry` extra | NVML board metrics via nvidia-ml-py |
| `agent` extra | The pinned Bub SDK integration |
| `quality` group | Ruff, ty, prek, and pytest |
| `docs` group | Sphinx, Furo, and MyST |
| `plot` group | Rendering saved evaluation results |

After changing dependencies, run `mise run lock`, review the lock diff, and sync
the relevant selections with `uv sync`. CI refuses a missing or stale
lock rather than resolving different versions. Avoid replacing a trained run's
environment while it is active.

## Checks and builds

`mise run check` verifies the lock, repository hygiene, Ruff lint and formatting,
and ty checks for `src` against the installed dependencies. `mise run lint` runs the
checks that only need the `quality` group. `uv run --group quality prek install`
optionally installs the same hygiene hooks locally; installing hooks is not
required to run the checks.

`mise run test` runs pytest, including the pinned RLCD loss and
gradient values, prefix-cache updates and gradients, calibration metrics, and
data-split isolation. It does not load
model weights, download data, or prove GPU training/inference acceptance. CI runs
these tests with Hugging Face access disabled. GPU acceptance remains a separate
step for changes to execution or training.

Run an individual file or case with `uv run pytest tests/test_rlcd.py` or
`uv run pytest -k gradients`. Test discovery is limited to `tests/` in
`pyproject.toml`.

The test job installs the same locked CUDA wheels, checks types, and runs the
tests on CPU. Quality and website jobs
do not install the model runtime.

`mise run build` uses `uv_build` to produce a wheel and source distribution in
`dist/`. Local model weights, datasets, caches, and experiment runs are excluded.
The CI build does not publish packages. Website validation and deployment use
the separate [website workflow](website.md).

The mise tasks and separated quality/test jobs adapt the relevant parts of
[Bub's CI layout](https://github.com/bubbuild/bub/blob/main/Makefile) and
[CI](https://github.com/bubbuild/bub/blob/main/.github/workflows/main.yml).
