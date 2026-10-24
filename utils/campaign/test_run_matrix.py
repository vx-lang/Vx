"""Unit tests for run_matrix.classify_outcome, one for each pattern of the pilot runs (#289).

    python3 -m unittest discover -s utils/campaign -p 'test_*.py'

The strings are the verbatim substrings that vLLM 0.31.0 printed in the pilot. The numbers in them
are the pilot's own examples.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matrix import classify_outcome  # noqa: E402

STARTED = "GPU KV cache size: 104624 tokens"


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


if __name__ == "__main__":
    unittest.main()
