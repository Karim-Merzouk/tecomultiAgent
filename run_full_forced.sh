#!/usr/bin/env bash
# Full 1470-sample (en+ar => 2940 case-runs) benchmark of the NO-ESCALATION
# (forced-completion) config: the multi-agent attempts 100% of cases like a
# single-model baseline. Waits for any in-flight run to free the GPU first.
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=src TELCO_EMBED=fallback

echo "QUEUED $(date) — waiting for GPU (any in-flight python cli/smoke) to free"
while pgrep -f 'cli run-eval' | grep -qv "$$" || pgrep -f 'smoke_forced.py' >/dev/null; do sleep 20; done
sleep 8

echo "FULLSTART $(date)"
python -m telco_multiagent.cli run-eval --config full_system_forced_real \
  --split all --languages en,ar --n-samples-per-blueprint 60 \
  --results-dir results_full_forced --no-traces 2>&1 \
  | tee results_full_forced_raw.log | grep -E 'cases=|consistency' \
  || echo "FAILED full_system_forced_real (see results_full_forced_raw.log)"

python -m telco_multiagent.cli report --results-dir results_full_forced \
  --out results_full_forced/REPORT.md >/dev/null 2>&1
echo "FULLDONE $(date)"
