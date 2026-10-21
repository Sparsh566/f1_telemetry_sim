"""
run_tests.py
============
Standalone test runner that executes all tests in tests/ with UTF-8 stdout and unbuffered flush.
"""

import sys
import os
import inspect
import traceback
import importlib

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_MODULES = [
    "tests.test_activations",
    "tests.test_losses",
    "tests.test_optimizers",
    "tests.test_backpropagation",
    "tests.test_model_persistence",
    "tests.test_ml_pipeline",
    "tests.test_perceptron_xor"
]

def main():
    print("=" * 70, flush=True)
    print("FORMULA 1 TELEMETRY SIMULATOR: TEST SUITE RUNNER", flush=True)
    print("=" * 70, flush=True)
    total_passed = 0
    total_failed = 0
    failures = []

    for mod_name in TEST_MODULES:
        print(f"\n[RUN] Running {mod_name}...", flush=True)
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[FAIL] Failed to import {mod_name}: {e}", flush=True)
            traceback.print_exc()
            total_failed += 1
            failures.append((mod_name, "import", str(e)))
            continue

        test_funcs = [
            (name, func) for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        for name, func in test_funcs:
            try:
                func()
                print(f"  [PASS] {name}", flush=True)
                total_passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}", flush=True)
                traceback.print_exc()
                total_failed += 1
                failures.append((mod_name, name, str(e)))

    print("\n" + "=" * 70, flush=True)
    print(f"RESULTS: {total_passed} PASSED, {total_failed} FAILED", flush=True)
    print("=" * 70, flush=True)

    if total_failed > 0:
        print("\nSummary of Failures:", flush=True)
        for mod, name, err in failures:
            print(f"  - {mod}.{name}: {err}", flush=True)
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY!", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    main()
