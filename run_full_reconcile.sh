#!/usr/bin/env bash
# Full 1470-sample (en+ar => 2940 case-runs) benchmark of the reconcile config.
# Waits for any in-flight run-eval (the v2 28-case run) to finish first so the
# single GPU is not contended, then runs the full split.
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=src TELCO_EMBED=fallback

echo "QUEUED $(date) — waiting for any in-flight run-eval to finish"
while pgrep -f 'cli run-eval' | grep -qv "$$"; do sleep 30; done
# small settle so ollama frees VRAM
sleep 10

echo "FULLSTART $(date)"
python -m telco_multiagent.cli run-eval --config full_system_reconcile_real \
  --split all --languages en,ar --n-samples-per-blueprint 60 \
  --results-dir results_full_reconcile --no-traces 2>&1 \
  | grep -E 'cases=|consistency' || echo "FAILED full_system_reconcile_real"

python -m telco_multiagent.cli report --results-dir results_full_reconcile \
  --out results_full_reconcile/REPORT.md >/dev/null 2>&1
echo "FULLDONE $(date)"
