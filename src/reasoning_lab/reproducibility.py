from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from typing import Any

import torch


PACKAGE_NAMES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "peft",
    "torch",
    "transformers",
    "trl",
)


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    gpu: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = None
    gpu["driver_version"] = driver

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def run_gpu_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QLoRA Stage 1 pipeline")
    if torch.cuda.get_device_capability(0) < (12, 0):
        raise RuntimeError("This configuration is validated for the local RTX 5090 (sm_120)")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured bfloat16 compute dtype is unavailable")

    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(
        128,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    ).to("cuda")
    inputs = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    outputs = layer(inputs)
    loss = outputs.float().square().mean()
    loss.backward()
    result = {
        "nf4_forward_shape": list(outputs.shape),
        "nf4_forward_finite": bool(torch.isfinite(outputs).all().item()),
        "nf4_backward_finite": bool(torch.isfinite(inputs.grad).all().item()),
    }
    del layer, inputs, outputs, loss
    torch.cuda.empty_cache()
    if not result["nf4_forward_finite"] or not result["nf4_backward_finite"]:
        raise RuntimeError("bitsandbytes NF4 CUDA preflight produced non-finite values")
    return result

