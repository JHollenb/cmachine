#!/usr/bin/env bash
# usage: run.sh NAME  (payload root = cwd)
set -euo pipefail
NAME=$1
RES=/mnt/usb-storage/experiment-results/cmachine-ledger
PY=/mnt/hdd8tb/uv-archive-stage/kS7-10fOwViqzafr/bin/python
mkdir -p "$RES" /mnt/usb-storage/tmp
export TMPDIR=/mnt/usb-storage/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 PATH=/usr/local/cuda/bin:$PATH
echo "=== $NAME $(date -u +%FT%TZ) $(hostname) job=${MRUN_JOB_ID:-}"
nvcc --version | tail -2 || true
exec "$PY" -u worker.py "$RES/${NAME}-${MRUN_JOB_ID:-nojob}" plan.json
