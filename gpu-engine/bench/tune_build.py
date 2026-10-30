#!/usr/bin/env python3
"""Build one PerfLens Tune candidate: config JSON -> nvcc flags -> kernel_bench binary.

Usage (from gpu-engine/):
    python3 bench/tune_build.py <kernel> '<config-json>' [--force]

Prints one JSON object on stdout:
    {"kernel", "config", "flags", "cmd", "ok", "returncode", "compile_ms",
     "binary", "ptxas_log", "cached", "error"}

`ptxas_log` is the full nvcc/ptxas stderr from the build (`-Xptxas -v`);
PerfLens parses registers/spills/smem out of it. `compile_ms` is the wall
time of the nvcc invocation. Builds are cached per (kernel, config, arch)
under build/tune/; a cached result reports the original compile_ms with
"cached": true. Pass --force to rebuild.

Environment: NVCC (default "nvcc"), TUNE_ARCH (default "sm_86").
Only the standard library is used, so this runs anywhere Python 3.9+ does.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ENGINE_DIR = Path(__file__).resolve().parent.parent

# kernel name -> (kernel source, bench define, macro prefix, supports blocks_per_row + vec)
KERNELS: dict[str, tuple[str, str, str, bool]] = {
    "bias_gelu": ("src/kernels/gelu.cu", "BENCH_GELU", "GELU", True),
    "add_layernorm": ("src/kernels/layernorm.cu", "BENCH_LAYERNORM", "LN", False),
    "argmax": ("src/kernels/logits.cu", "BENCH_ARGMAX", "ARGMAX", False),
}

CONFIG_KEYS = {"threads", "blocks_per_row", "vec", "maxrregcount", "fast_math", "min_blocks"}
MAX_THREADS_PER_SM = 1536  # sm_86; mirrors include/engine/tune_params.h


class ConfigError(ValueError):
    pass


def normalize(kernel: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate a candidate config and fill defaults. Raises ConfigError."""
    if kernel not in KERNELS:
        raise ConfigError(f"unknown kernel {kernel!r}; expected one of {sorted(KERNELS)}")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    _, _, _, has_bpr = KERNELS[kernel]

    threads = config.get("threads", 256)
    if not isinstance(threads, int) or threads < 32 or threads > 1024 or threads & (threads - 1):
        raise ConfigError(f"threads must be a power of two in [32, 1024], got {threads!r}")

    bpr = config.get("blocks_per_row", 1)
    if not isinstance(bpr, int) or bpr < 1:
        raise ConfigError(f"blocks_per_row must be an int >= 1, got {bpr!r}")
    if bpr != 1 and not has_bpr:
        raise ConfigError(f"{kernel} does not support blocks_per_row != 1")

    vec = config.get("vec", 1)
    if vec not in (1, 2) or isinstance(vec, bool):
        raise ConfigError(f"vec must be 1 or 2, got {vec!r}")
    if vec != 1 and not has_bpr:
        raise ConfigError(f"{kernel} does not support vec != 1")

    maxrreg = config.get("maxrregcount")
    if maxrreg is not None and (not isinstance(maxrreg, int) or maxrreg < 16 or maxrreg > 255):
        raise ConfigError(f"maxrregcount must be null or an int in [16, 255], got {maxrreg!r}")

    fast_math = config.get("fast_math", False)
    if not isinstance(fast_math, bool):
        raise ConfigError(f"fast_math must be a bool, got {fast_math!r}")

    min_blocks = config.get("min_blocks")
    if min_blocks is not None:
        if not isinstance(min_blocks, int) or min_blocks < 1:
            raise ConfigError(f"min_blocks must be null or an int >= 1, got {min_blocks!r}")
        if threads * min_blocks > MAX_THREADS_PER_SM:
            raise ConfigError(
                f"threads*min_blocks = {threads * min_blocks} exceeds {MAX_THREADS_PER_SM} threads/SM"
            )

    out: dict[str, Any] = {
        "threads": threads,
        "maxrregcount": maxrreg,
        "fast_math": fast_math,
        "min_blocks": min_blocks,
    }
    if has_bpr:
        out["blocks_per_row"] = bpr
        out["vec"] = vec
    return out


def flags_for(kernel: str, config: dict[str, Any]) -> list[str]:
    """nvcc flags for a candidate (excluding the fixed build flags)."""
    cfg = normalize(kernel, config)
    _, _, prefix, has_bpr = KERNELS[kernel]
    flags = [f"-D{prefix}_THREADS={cfg['threads']}"]
    if has_bpr:
        flags.append(f"-D{prefix}_BLOCKS_PER_ROW={cfg['blocks_per_row']}")
        flags.append(f"-D{prefix}_VEC={cfg['vec']}")
    if cfg["min_blocks"] is not None:
        flags.append(f"-D{prefix}_MIN_BLOCKS={cfg['min_blocks']}")
    if cfg["maxrregcount"] is not None:
        flags.append(f"-maxrregcount={cfg['maxrregcount']}")
    if cfg["fast_math"]:
        flags.append("--use_fast_math")
    return flags


def cache_key(kernel: str, config: dict[str, Any], arch: str) -> str:
    cfg = normalize(kernel, config)
    blob = json.dumps({"kernel": kernel, "config": cfg, "arch": arch}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def build_command(kernel: str, config: dict[str, Any], out_binary: Path, nvcc: str, arch: str) -> list[str]:
    src, bench_define, _, _ = KERNELS[kernel]
    return [
        nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        "-Iinclude",
        "-Isrc",
        "-Xptxas",
        "-v",
        f"-D{bench_define}",
        *flags_for(kernel, config),
        "bench/kernel_bench.cu",
        src,
        "-o",
        str(out_binary),
    ]


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def build(kernel: str, config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    nvcc = os.environ.get("NVCC", "nvcc")
    arch = os.environ.get("TUNE_ARCH", "sm_86")
    cfg = normalize(kernel, config)
    out_dir = ENGINE_DIR / "build" / "tune" / f"{kernel}-{cache_key(kernel, cfg, arch)}"
    binary = out_dir / "kernel_bench"
    record_path = out_dir / "build.json"

    if not force and record_path.exists() and binary.exists():
        record = json.loads(record_path.read_text())
        if record.get("ok"):
            record["cached"] = True
            return record

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(kernel, cfg, binary, nvcc, arch)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ENGINE_DIR, capture_output=True, text=True, timeout=600)
        returncode, stderr = proc.returncode, proc.stdout + proc.stderr
    except FileNotFoundError:
        returncode, stderr = 127, f"compiler not found: {nvcc}"
    except subprocess.TimeoutExpired:
        returncode, stderr = 124, "nvcc timed out after 600s"
    compile_ms = (time.perf_counter() - t0) * 1000.0

    log_path = out_dir / "ptxas.log"
    log_path.write_text(stderr)
    ok = returncode == 0 and binary.exists()
    record = {
        "kernel": kernel,
        "config": cfg,
        "flags": flags_for(kernel, cfg),
        "cmd": cmd,
        "ok": ok,
        "returncode": returncode,
        "compile_ms": round(compile_ms, 1),
        "binary": str(binary),
        "ptxas_log": str(log_path),
        "cached": False,
        "error": None if ok else _last_line(stderr) or f"build failed (returncode {returncode})",
    }
    record_path.write_text(json.dumps(record, indent=2))
    return record


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--force"]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    kernel, raw = args
    try:
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise ConfigError("config must be a JSON object")
        record = build(kernel, config, force="--force" in argv)
    except (ConfigError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": f"invalid config: {exc}"}))
        return 2
    print(json.dumps(record))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
