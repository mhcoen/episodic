#!/bin/bash
# Ship Check - Run before merging/deploying changes
#
# This script runs all critical verification checks:
# 1. Reactivation tests (feature correctness)
# 2. Benchmark tests (performance regression)
# 3. Phase 3 verification (invariants)
#
# Exit code: 0 = all passed, 1 = failure

set -e

echo "=============================================="
echo "Ship Check"
echo "=============================================="
echo ""

export EPISODIC_TEST_MODE=1

echo "1. Running reactivation tests..."
echo "----------------------------------------------"
pytest -m reactivation -v
echo ""

echo "2. Running benchmark tests..."
echo "----------------------------------------------"
pytest tests/benchmark/ -v
echo ""

echo "3. Running Phase 3 verification..."
echo "----------------------------------------------"
python -m episodic.maintenance.verify_phase3
echo ""

echo "=============================================="
echo "All checks passed! Ready to ship."
echo "=============================================="
