#!/bin/bash
# One Stage-2 epoch eval on alibaba (1 PPU card, queue brain), submitted by h200/alibaba/poll.sh
# after it claimed the epoch on the S3 board (h200/eval_dispatch.py). Paths come from SITE_ENV.
#SBATCH --gres=gpu:brain:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J lsft-eval
set -euo pipefail
: "${EP:?}" "${SITE_ENV:?}"
. "$SITE_ENV"
set -a; . "$HOME/.s3-cod.env"; set +a
export PYTHONPATH="$PYLIB" PYTHONIOENCODING=utf-8
echo "== job $SLURM_JOB_ID epoch $EP $(date)"
# No --bind /dev: setuid apptainer already mounts the host /dev (PPU visible); binding it again
# inside a Slurm job broke /dev/urandom ("Unable to open /dev/urandom" on import torch).
apptainer exec --bind /bmcp_lvm_fs "$SIF" python3 "$REPO/h200/eval_dispatch.py" run \
  --base "$BASE" --site alibaba --epoch "$EP" --model "$MODEL" --init_prefix "$INIT_PREFIX" \
  --data "$DATA" --work "$WORK" --batch "${EVAL_BATCH:-64}" ${DISPATCH_ARGS:-}
