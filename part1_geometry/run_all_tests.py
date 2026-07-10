#!/usr/bin/env python3
"""
Run all Part 1 tests.
Exit 0: all pass.
Exit 1: one or more test failures.
Exit 2: runner infrastructure error (directory missing, no test files, etc.)

Mirrors the fix applied to Part 2's test runner (audit finding D1).
Never silently reports 0/0.
"""
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent / "tests"

def main() -> int:
    if not TESTS_DIR.exists():
        print(f"ERROR: tests/ directory not found at {TESTS_DIR}", file=sys.stderr)
        print("Create the tests/ directory and add test_*.py files.", file=sys.stderr)
        return 2

    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    if not test_files:
        print(f"ERROR: No test_*.py files found in {TESTS_DIR}", file=sys.stderr)
        return 2

    passed = 0
    failed = 0

    for tf in test_files:
        if not tf.is_file():
            print(f"SKIP {tf.name} (not a file)")
            continue

        result = subprocess.run(
            [sys.executable, str(tf)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            passed += 1
            print(f"PASS {tf.name}")
        else:
            failed += 1
            print(f"FAIL {tf.name}")
            if result.stdout.strip():
                print(result.stdout)
            if result.stderr.strip():
                print(result.stderr, file=sys.stderr)

    total = passed + failed
    print(f"\nTOTAL: {passed} passed, {failed} failed ({total} test files)")

    if total == 0:
        print("ERROR: No test files ran.", file=sys.stderr)
        return 2

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())