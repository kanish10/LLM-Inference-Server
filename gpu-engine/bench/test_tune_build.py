"""CPU-only tests for bench/tune_build.py (no CUDA toolchain or GPU needed).

    python3 -m unittest discover -s bench -p "test_*.py"
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_build  # noqa: E402

FAKE_NVCC = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
out = args[args.index("-o") + 1]
if any(a.startswith("-DGELU_THREADS=64") for a in args):
    sys.stderr.write("fake error: refusing 64 threads\\n")
    sys.exit(1)
sys.stderr.write("ptxas info    : Used 14 registers, used 0 barriers, 380 bytes cmem[0]\\n")
open(out, "w").write("binary")
"""


class NormalizeTest(unittest.TestCase):
    def test_defaults_match_shipped_launch_config(self) -> None:
        self.assertEqual(
            tune_build.normalize("bias_gelu", {}),
            {
                "threads": 256,
                "blocks_per_row": 1,
                "vec": 1,
                "maxrregcount": None,
                "fast_math": False,
                "min_blocks": None,
            },
        )
        self.assertNotIn("blocks_per_row", tune_build.normalize("argmax", {}))

    def test_rejects_invalid_configs(self) -> None:
        bad = [
            ("bias_gelu", {"threads": 96}),  # not a power of two
            ("bias_gelu", {"threads": 2048}),
            ("bias_gelu", {"blocks_per_row": 0}),
            ("bias_gelu", {"vec": 4}),
            ("bias_gelu", {"vec": True}),
            ("add_layernorm", {"blocks_per_row": 2}),  # only GELU splits rows
            ("argmax", {"vec": 2}),
            ("argmax", {"threads": 1024, "min_blocks": 2}),  # 2048 threads > one SM
            ("argmax", {"maxrregcount": 8}),
            ("argmax", {"fast_math": 1}),
            ("argmax", {"unroll": 4}),
            ("softmax", {}),
        ]
        for kernel, cfg in bad:
            with self.subTest(kernel=kernel, cfg=cfg):
                with self.assertRaises(tune_build.ConfigError):
                    tune_build.normalize(kernel, cfg)


class FlagsTest(unittest.TestCase):
    def test_gelu_flags(self) -> None:
        cfg = {
            "threads": 512,
            "blocks_per_row": 4,
            "vec": 2,
            "fast_math": True,
            "min_blocks": 2,
            "maxrregcount": 32,
        }
        self.assertEqual(
            tune_build.flags_for("bias_gelu", cfg),
            [
                "-DGELU_THREADS=512",
                "-DGELU_BLOCKS_PER_ROW=4",
                "-DGELU_VEC=2",
                "-DGELU_MIN_BLOCKS=2",
                "-maxrregcount=32",
                "--use_fast_math",
            ],
        )

    def test_layernorm_and_argmax_flags_use_their_own_prefix(self) -> None:
        self.assertEqual(tune_build.flags_for("add_layernorm", {"threads": 128}), ["-DLN_THREADS=128"])
        self.assertEqual(
            tune_build.flags_for("argmax", {"threads": 1024, "fast_math": True}),
            ["-DARGMAX_THREADS=1024", "--use_fast_math"],
        )

    def test_cache_key_ignores_key_order_and_explicit_defaults(self) -> None:
        a = tune_build.cache_key("bias_gelu", {"threads": 128, "vec": 2}, "sm_86")
        b = tune_build.cache_key("bias_gelu", {"vec": 2, "threads": 128, "fast_math": False}, "sm_86")
        c = tune_build.cache_key("bias_gelu", {"threads": 128, "vec": 2}, "sm_89")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_command_builds_one_kernel_tu_with_bench(self) -> None:
        cmd = tune_build.build_command("argmax", {}, Path("/tmp/x"), "nvcc", "sm_86")
        self.assertIn("-DBENCH_ARGMAX", cmd)
        self.assertIn("src/kernels/logits.cu", cmd)
        self.assertNotIn("src/kernels/gelu.cu", cmd)
        self.assertEqual(cmd[cmd.index("-Xptxas") + 1], "-v")


class BuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        nvcc = root / "fake_nvcc"
        nvcc.write_text(FAKE_NVCC)
        nvcc.chmod(nvcc.stat().st_mode | stat.S_IEXEC)
        self.env = mock.patch.dict(os.environ, {"NVCC": str(nvcc), "TUNE_ARCH": "sm_86"})
        self.env.start()
        self.engine = mock.patch.object(tune_build, "ENGINE_DIR", root)
        self.engine.start()

    def tearDown(self) -> None:
        self.engine.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_build_records_log_timing_and_caches(self) -> None:
        first = tune_build.build("bias_gelu", {"threads": 128})
        self.assertTrue(first["ok"])
        self.assertFalse(first["cached"])
        self.assertGreaterEqual(first["compile_ms"], 0.0)
        self.assertIn("Used 14 registers", Path(first["ptxas_log"]).read_text())
        self.assertTrue(Path(first["binary"]).exists())

        second = tune_build.build("bias_gelu", {"threads": 128})
        self.assertTrue(second["cached"])
        self.assertEqual(second["compile_ms"], first["compile_ms"])

        forced = tune_build.build("bias_gelu", {"threads": 128}, force=True)
        self.assertFalse(forced["cached"])

    def test_failed_build_reports_last_compiler_line(self) -> None:
        rec = tune_build.build("bias_gelu", {"threads": 64})
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["returncode"], 1)
        self.assertEqual(rec["error"], "fake error: refusing 64 threads")
        # A failed build is never served from cache.
        self.assertFalse(tune_build.build("bias_gelu", {"threads": 64})["cached"])

    def test_main_rejects_invalid_config_without_building(self) -> None:
        with mock.patch("builtins.print") as printed:
            rc = tune_build.main(["argmax", json.dumps({"threads": 100})])
        self.assertEqual(rc, 2)
        out = json.loads(printed.call_args.args[0])
        self.assertFalse(out["ok"])
        self.assertIn("power of two", out["error"])


if __name__ == "__main__":
    unittest.main()
