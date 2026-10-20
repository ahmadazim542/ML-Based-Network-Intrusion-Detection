#!/usr/bin/env bash
# Full overnight run: every step on the random split, then on the time-block split.
#
#   nohup scripts/run_full.sh > /dev/null 2>&1 &
#
# - caffeinate keeps the Mac awake while it runs (keep it plugged in; closing the lid
#   still sleeps most MacBooks).
# - Each split logs to results/full_<split>.log; results/full_run.status says how it ended.
# - The cross-dataset step fails cleanly until the 2018 data is fetched
#   (scripts/fetch_data.py cicids2018); everything else still runs. Re-run it later with
#   `ids cross` and `ids cross --split time_block`.
set -u
cd "$(dirname "$0")/.."
STATUS=results/full_run.status
mkdir -p results
echo "running since $(date '+%F %T')" > "$STATUS"

for split in random time_block; do
    start=$(date +%s)
    caffeinate -i -s .venv/bin/ids all --split "$split" > "results/full_${split}.log" 2>&1
    code=$?
    mins=$(( ($(date +%s) - start) / 60 ))
    echo "$split: exit $code after ${mins} min ($(date '+%F %T'))" >> "$STATUS"
done
echo "finished $(date '+%F %T')" >> "$STATUS"
