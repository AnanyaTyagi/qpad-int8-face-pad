"""TensorRT engine building, inference and latency benchmarking (TensorRT 10.x API).

TensorRT is imported lazily so the rest of the package works without it.
All buffers are PyTorch CUDA tensors (no PyCUDA dependency).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .reproducibility import bytes_sha256, gpu_info, package_versions, write_json


def require_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("TensorRT is not installed. In Colab/Modal run: pip install tensorrt-cu13==10.16.1.11 (see configs/int8_experiments.yaml)") from exc
    return trt


def hardware_fingerprint() -> dict[str, Any]:
    """GPU name, driver and library versions identifying a latency environment."""
    info = gpu_info()
    vers = package_versions(("torch", "tensorrt", "onnx", "onnxruntime"))
    return {"gpu_name": info.get("gpu_name"), "driver_version": info.get("driver_version"),
            "torch_cuda": info.get("torch_cuda"), "cudnn": info.get("cudnn"), "tensorrt": vers.get("tensorrt"),
            "torch": vers.get("torch")}


def assert_same_hardware(fingerprints: list[Mapping[str, Any]]) -> None:
    """Refuse to combine latency results measured on different GPUs or software stacks."""
    keys = ("gpu_name", "driver_version", "tensorrt", "torch_cuda")
    distinct = {tuple(f.get(k) for k in keys) for f in fingerprints}
    if len(distinct) > 1:
        raise RuntimeError(f"latency results come from {len(distinct)} different environments: {distinct}. "
                           "They are not comparable; re-measure on one fixed GPU.")


def build_engine(onnx_path: Path, engine_path: Path, precision: str, tf32: bool, trt_cfg: Mapping[str, Any],
                 calibrator=None, log_path: Path | None = None) -> dict[str, Any]:
    """Build a serialized engine and write ``<engine>.json`` metadata. Returns the metadata.

    ``precision`` is ``fp32``, ``fp16`` or ``int8``. For strict FP32 checks pass
    ``tf32=False`` (TF32 is cleared explicitly). Every other builder flag is identical
    across precisions.
    """
    trt = require_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + "; ".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(float(trt_cfg["workspace_gb"]) * (1 << 30)))
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if calibrator is None:
            raise ValueError("INT8 build requires a calibrator")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision}")
    inp = network.get_input(0)
    size = tuple(inp.shape[1:])
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (int(trt_cfg["min_batch"]), *size), (int(trt_cfg["opt_batch"]), *size),
                      (int(trt_cfg["max_batch"]), *size))
    config.add_optimization_profile(profile)
    if precision == "int8":
        config.set_calibration_profile(profile)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see log)")
    data = bytes(serialized)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(data)
    meta = {"engine": engine_path.name, "precision": precision, "tf32": tf32, "sha256": bytes_sha256(data),
            "size_bytes": len(data), "build_seconds": time.time() - t0, "onnx": str(onnx_path),
            "trt_config": dict(trt_cfg), "hardware": hardware_fingerprint(),
            "calibrator": type(calibrator).__name__ if calibrator is not None else None}
    write_json(engine_path.with_suffix(engine_path.suffix + ".json"), meta)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta


class TrtRunner:
    """Execute a serialized engine on CUDA tensors; returns the pixel-map output."""

    def __init__(self, engine_path: Path, output_name: str = "pixel_map") -> None:
        import torch

        trt = require_tensorrt()
        self.trt, self.torch = trt, torch
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.output_name = output_name
        self.stream = torch.cuda.Stream()

    def infer_torch(self, x):
        """Run on a CUDA float32 tensor; returns a dict of output tensors."""
        torch = self.torch
        x = x.contiguous()
        self.context.set_input_shape(self.inputs[0], tuple(x.shape))
        self.context.set_tensor_address(self.inputs[0], x.data_ptr())
        outs = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            outs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name, outs[name].data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return outs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()
        return self.infer_torch(t)[self.output_name].cpu().numpy()


def benchmark(runner: TrtRunner, batch: int, size: int, warmup: int, iters: int) -> dict[str, Any]:
    """GPU latency per batch using CUDA events (excludes host<->device copies and preprocessing)."""
    torch = runner.torch
    x = torch.zeros((batch, 3, size, size), dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        runner.infer_torch(x)
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(runner.stream)
        runner.infer_torch(x)
        end.record(runner.stream)
        end.synchronize()
        times.append(start.elapsed_time(end))
    t = np.array(times)
    return {"batch": batch, "warmup": warmup, "iters": iters, "median_ms": float(np.median(t)),
            "p90_ms": float(np.percentile(t, 90)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "throughput_img_s": float(batch / (np.median(t) / 1000.0)),
            "hardware": hardware_fingerprint(),
            "scope": "GPU execute_async_v3 only; excludes decode, alignment and host-device copies"}
