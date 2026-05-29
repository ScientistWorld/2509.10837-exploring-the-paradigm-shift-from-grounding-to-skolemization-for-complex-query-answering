#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after generation: < 5 MB
#
# This benchmark is generated locally and does not require external files.

set -e

cd /home/user

if [ -f data/synthetic/train.jsonl ] && [ -f data/synthetic/valid.jsonl ] && [ -f data/synthetic/test.jsonl ] && [ -f data/synthetic/meta.json ]; then
  echo "Synthetic CQA data already exists; skipping generation."
else
  python data/synthetic_cqa.py --out data/synthetic --entities 160 --train 3000 --valid 500 --test 700 --seed 7
fi
