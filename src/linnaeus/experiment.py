"""Raw latency, environment, and resource measurements for reproducible runs."""

import importlib.metadata
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import torch

from linnaeus.model import DecisionModel, model_revision

GIB = 1024**3


def detect_gpu_device() -> Path:
    """Locate the sysfs DRM device that exposes GPU telemetry.

    Upstream hardcoded /sys/class/drm/card1/device, which only matches the
    author's AMD desktop. Honor LINNAEUS_GPU_DEVICE first, then scan for a
    card exposing AMD's mem_info_vram_used. NVIDIA hosts usually have no such
    card; the returned nonexistent path makes Sampler fall back to NVML/RSS.
    """
    override = os.environ.get("LINNAEUS_GPU_DEVICE")
    if override:
        return Path(override)
    for device in sorted(Path("/sys/class/drm").glob("card*/device")):
        if (device / "mem_info_vram_used").exists():
            return device
    return Path("/sys/class/drm/card0/device")


def _nvml_handle(index: int = 0):
    """Return an initialized NVML device handle, or None when unavailable."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(
            int(os.environ.get("LINNAEUS_GPU_INDEX", index))
        )
    except Exception:
        return None


class Sampler:
    """Sample board memory/utilization/power and process RSS every 20 ms.

    AMD boards are read from sysfs; on NVIDIA hosts the same field names and
    units are served through NVML when nvidia-ml-py is installed.
    """

    def __init__(self, device: Path, interval: float = 0.02, *, output=None):
        self.device = device
        self.interval = interval
        self.output = output
        self.stream = None
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        hwmons = list((device / "hwmon").glob("hwmon*"))
        self.hwmon = hwmons[0] if hwmons else None
        self.nvml = None if (device / "mem_info_vram_used").exists() else _nvml_handle()

    def read(self):
        values = {}
        fields = {
            "board_vram_bytes": self.device / "mem_info_vram_used",
            "gpu_busy_percent": self.device / "gpu_busy_percent",
        }
        if self.hwmon:
            fields.update(
                {
                    "power_microwatts": self.hwmon / "power1_average",
                    "temperature_millidegrees": self.hwmon / "temp1_input",
                }
            )
        for name, path in fields.items():
            if path.exists():
                values[name] = int(path.read_text().strip())
        if self.nvml and "board_vram_bytes" not in values:
            pynvml, handle = self.nvml
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                values["board_vram_bytes"] = info.used
                values["gpu_busy_percent"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                values["power_microwatts"] = pynvml.nvmlDeviceGetPowerUsage(handle) * 1000
                values["temperature_millidegrees"] = (
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) * 1000
                )
            except Exception:
                pass
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                values["rss_bytes"] = int(line.split()[1]) * 1024
                break
        return values

    def run(self):
        while not self.stop.is_set():
            row = self.read()
            self.samples.append(row)
            if self.stream:
                self.stream.write(json.dumps({"unix_time": time.time(), **row}) + "\n")
            self.stop.wait(self.interval)

    def __enter__(self):
        if self.output:
            self.stream = self.output.open("a", buffering=1)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        if self.stream:
            self.stream.close()

    def summary(self):
        keys = {key for row in self.samples for key in row}
        return {
            key: {"max": max(values), "mean": statistics.mean(values)}
            for key in sorted(keys)
            if (values := [row[key] for row in self.samples if key in row])
        }


def emit(path: Path, row: dict):
    encoded = json.dumps(row, ensure_ascii=False)
    with path.open("a") as stream:
        stream.write(encoded + "\n")
    print(encoded, flush=True)


def latency_stats(durations: list[float], batch: int):
    ordered = sorted(durations)
    # Linear interpolation (NumPy's default definition), including short runs.
    index = 0.95 * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {
        "p50_ms": statistics.median(durations),
        "p95_ms": p95,
        "percentile_method": "linear interpolation, index q*(n-1)",
        "mean_ms": statistics.mean(durations),
        "samples_ms": durations,
        "decisions_per_second": 1000 * batch / statistics.mean(durations),
        "amortized_ms_per_decision": statistics.mean(durations) / batch,
    }


def timed(operation, repeats: int):
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return durations


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "device_free_gib": free / GIB,
        "device_total_gib": total / GIB,
    }


def environment(model: DecisionModel, checkpoint: Path):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    return {
        "kind": "environment",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: _package_version(name)
            for name in [
                "torch",
                "torchvision",
                "transformers",
                "peft",
                "pytorch-triton-rocm",
                "triton",
                "flash-linear-attention",
                "fla-core",
            ]
        },
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "gpu_properties": str(torch.cuda.get_device_properties(0)),
        "checkpoint_revision": model_revision(checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
        "head_parameters": sum(p.numel() for p in model.head.parameters()),
        "attention": "sdpa",
        "dtype": str(next(model.backbone.parameters()).dtype),
        "kernel_flags": {
            "linear_patch": True,
            "triton_convolution": True,
            "fused_norm_and_swiglu": True,
            "shared_prefix": True,
            "frozen_vision_cache_MiB": 128,
            "experimental_rocm_sdpa": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
            "delta_rule": str(implementation.torch_chunk_gated_delta_rule),
        },
        "threads": torch.get_num_threads(),
        "memory": memory(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("ROCR_VISIBLE_DEVICES"),
    }


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
