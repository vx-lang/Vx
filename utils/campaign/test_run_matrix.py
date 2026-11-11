"""Unit tests for run_matrix: classify_outcome, one test for each pattern of the pilot runs
(#289), and run_engine, with fake_vllm.py in place of vLLM.

    python3 -m unittest discover -s utils/campaign -p 'test_*.py'

The strings are the verbatim substrings that vLLM 0.31.0 printed in the pilot. The numbers in them
are the pilot's own examples.
"""

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_matrix import classify_outcome, run_engine  # noqa: E402

STARTED = "GPU KV cache size: 104624 tokens"
SERVED = '"POST /v1/completions HTTP/1.1" 200 OK'
# The first line of vLLM's stop, and the error it can print during the stop.
SHUTDOWN = "[core.py:1409] [shutdown] EngineCore: trigger received signal=SIGINT"
ENGINE_DEAD = "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue."


class ClassifyOutcome(unittest.TestCase):
    def check(self, log, label, pattern, served=None, setup_fault=False):
        outcome = classify_outcome(log, served)
        self.assertEqual(outcome.label, label)
        self.assertEqual(outcome.pattern, pattern)
        self.assertEqual(outcome.setup_fault, setup_fault)

    def test_kv_cache_size_then_generation_is_boot_ok_serve_ok(self):
        self.check(STARTED + "\n", "BOOT_OK_SERVE_OK", "GPU KV cache size:", served=True)

    def test_estimated_maximum_model_length_is_engine_refuses(self):
        # Qwen2.5-32B, max_model_len 32768, on an H100 at 0.90 and 0.92.
        log = "estimated maximum model length is 25728\n"
        self.check(log, "ENGINE_REFUSES", "estimated maximum model length is")

    def test_cuda_out_of_memory_after_model_loading_is_oom_at_load(self):
        # Qwen2.5-72B on an H200: the weights load, then the profiling run runs out.
        log = ("Model loading took 135.43 GiB\n"
               "torch.OutOfMemoryError: CUDA out of memory\n"
               "Tried to allocate 924.00 MiB ... 753.25 MiB is free\n")
        self.check(log, "OOM_AT_LOAD", "torch.OutOfMemoryError: CUDA out of memory")

    def test_cuda_error_memory_allocation_after_startup_is_oom_at_prefill(self):
        # Qwen2.5-32B, a 32k prompt, on an H100 at 0.98 and 0.99.
        log = STARTED + "\ncudaErrorMemoryAllocation\n"
        self.check(log, "OOM_AT_PREFILL", "cudaErrorMemoryAllocation", served=False)

    def test_engine_dead_error_after_startup_is_oom_at_prefill(self):
        log = STARTED + "\nEngineDeadError\n"
        self.check(log, "OOM_AT_PREFILL", "EngineDeadError", served=False)

    def test_out_of_memory_after_startup_is_not_a_load_failure(self):
        # The order decides: after "GPU KV cache size", the engine's death is at prefill.
        log = STARTED + "\ntorch.OutOfMemoryError: CUDA out of memory\nEngineDeadError\n"
        self.check(log, "OOM_AT_PREFILL", "EngineDeadError", served=False)

    def test_engine_dead_error_while_vllm_stops_after_a_served_request_is_boot_ok_serve_ok(self):
        # Qwen2.5-32B, a 32k prompt, on an A100 80GB: the request came back, and vLLM printed
        # the error only after the harness stopped it.
        log = "\n".join([STARTED, SERVED, SHUTDOWN, ENGINE_DEAD]) + "\n"
        self.check(log, "BOOT_OK_SERVE_OK", "GPU KV cache size:", served=True)

    def test_engine_dead_error_before_vllm_stops_is_oom_at_prefill(self):
        log = "\n".join([STARTED, ENGINE_DEAD, SHUTDOWN]) + "\n"
        self.check(log, "OOM_AT_PREFILL", "EngineDeadError", served=True)

    def test_engine_dead_error_while_vllm_stops_after_a_failed_request_is_oom_at_prefill(self):
        # vLLM also prints [shutdown] lines when it stops after its own failure. So when the
        # request failed, an error after them still counts.
        log = "\n".join([STARTED, SHUTDOWN, ENGINE_DEAD]) + "\n"
        self.check(log, "OOM_AT_PREFILL", "EngineDeadError", served=False)

    def test_no_nvcc_is_a_setup_fault(self):
        log = "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist\n"
        self.check(log, "OTHER_FAIL",
                   "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist",
                   setup_fault=True)

    def test_no_nvcc_after_startup_is_still_a_setup_fault(self):
        # FlashInfer's sampler needs nvcc on the first request, so the engine can die after
        # startup. Its death must not count as an out-of-memory error at prefill.
        log = (STARTED + "\nCould not find nvcc and default cuda_home='/usr/local/cuda' "
               "doesn't exist\nEngineDeadError\n")
        self.check(log, "OTHER_FAIL",
                   "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist",
                   served=False, setup_fault=True)

    def test_no_c_compiler_is_a_setup_fault(self):
        log = "InductorError: RuntimeError: Failed to find C compiler\n"
        self.check(log, "OTHER_FAIL", "InductorError: RuntimeError: Failed to find C compiler",
                   setup_fault=True)

    def test_a_failed_request_with_no_pattern_is_other_fail(self):
        self.check(STARTED + "\n", "OTHER_FAIL", None, served=False)

    def test_startup_with_no_request_is_unclassified(self):
        self.check(STARTED + "\n", "UNCLASSIFIED", None)

    def test_a_log_with_no_pattern_is_unclassified(self):
        self.check("Segmentation fault\n", "UNCLASSIFIED", None)



class RunEngine(unittest.TestCase):
    """run_engine launches the engine, sends the cell's requests and stops it. The log holds
    the engine's stdout and stderr, and the classifier reads it."""

    def launch(self, mode):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        log = Path(tempfile.mkdtemp()) / "cell.log"
        os.environ["FAKE_VLLM"] = mode
        command = [sys.executable, str(HERE / "fake_vllm.py"), "serve", "model", "--port", str(port)]
        run = run_engine(command, log, port, "model", ctx=64, batch=2,
                         startup_timeout=60, request_timeout=60)
        self.assertIsNotNone(run["returncode"], "the engine must be stopped")
        return run, log.read_text()

    def test_an_engine_that_serves(self):
        run, log = self.launch("serves")
        self.assertTrue(run["served"])
        self.assertIn("GPU KV cache size: 1000 tokens", log)
        self.assertEqual(classify_outcome(log, run["served"]).label, "BOOT_OK_SERVE_OK")

    def test_an_engine_that_runs_out_of_memory_at_load(self):
        run, log = self.launch("oom_at_load")
        self.assertIsNone(run["served"])
        self.assertIn("Model loading took 135.43 GiB", log)
        self.assertIn("torch.OutOfMemoryError: CUDA out of memory", log)
        self.assertEqual(classify_outcome(log, run["served"]).label, "OOM_AT_LOAD")

    def test_an_engine_that_dies_on_the_first_request(self):
        run, log = self.launch("dies_on_request")
        self.assertFalse(run["served"])
        self.assertEqual(classify_outcome(log, run["served"]).label, "OOM_AT_PREFILL")

if __name__ == "__main__":
    unittest.main()
