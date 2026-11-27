"""Tests for gen_error_index.py: what it does when mdformat is missing or fails.

    python3 -m unittest discover -s scripts/tools -p 'test_*.py'

The tests run the script as a user would, from the repository root. They need only python3.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "tools" / "gen_error_index.py"


def run_check(path_dir):
    """Run `gen_error_index.py --check` with `path_dir` as the only directory on PATH."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        env={**os.environ, "PATH": str(path_dir)},
        capture_output=True,
        text=True,
    )


class MissingMdformat(unittest.TestCase):
    def test_check_says_mdformat_is_missing_and_does_not_say_the_page_is_old(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            result = run_check(empty_dir)
        self.assertEqual(result.returncode, 1)
        self.assertIn("mdformat is not installed", result.stderr)
        self.assertNotIn("out of date", result.stderr)

    def test_check_reports_an_mdformat_that_exits_with_an_error(self):
        with tempfile.TemporaryDirectory() as bin_dir:
            fake = Path(bin_dir) / "mdformat"
            fake.write_text("#!/bin/sh\necho 'bad input' >&2\nexit 2\n")
            fake.chmod(0o755)
            result = run_check(bin_dir)
        self.assertEqual(result.returncode, 1)
        self.assertIn("mdformat failed", result.stderr)
        self.assertIn("bad input", result.stderr)
        self.assertNotIn("out of date", result.stderr)


if __name__ == "__main__":
    unittest.main()
