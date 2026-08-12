#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=src TELCO_EMBED=fallback
echo "START $(date)"
for c in single_agent_real no_checker_real full_system_real; do
  echo "=== $c $(date +%H:%M:%S) ==="
  python -m telco_multiagent.cli run-eval --config $c --split test \
    --languages en,ar --n-samples-per-blueprint 1 --results-dir results_real --no-traces \
    2>&1 | grep -E 'cases=|consistency' || echo "FAILED $c"
done
python -m telco_multiagent.cli report --results-dir results_real --out results_real/REPORT.md >/dev/null 2>&1
echo "DONE $(date)"
