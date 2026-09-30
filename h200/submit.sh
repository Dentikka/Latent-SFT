#!/bin/bash
# Usage: bash h200/submit.sh <encoder|decoder|union> <preset> <exp-name> [KEY=VALUE ...]
# Creates exps/<series>/<exp-name>/ with run.env (preset + overrides) and cmd.txt,
# then submits h200/stage1.sbatch. Re-running on an existing exp resubmits it
# (resuming from its newest checkpoint) without touching run.env.
set -euo pipefail
source "$(dirname "$0")/env.sh"
STAGE="$1"; PRESET="$2"; EXP="$3"; shift 3
EXP_DIR="$LSFT_EXPS/$EXP"
mkdir -p "$EXP_DIR" "$LSFT_LOGS"
if [ ! -f "$EXP_DIR/run.env" ]; then
  cp "$LSFT_REPO/h200/presets/$PRESET.env" "$EXP_DIR/run.env"
  for kv in "$@"; do echo "$kv" >> "$EXP_DIR/run.env"; done
fi
rm -f "$EXP_DIR/STALLED" "$EXP_DIR/ckpt_at_start"
echo "$(date '+%F %T') bash h200/submit.sh $STAGE $PRESET $EXP $* @ $(git -C "$LSFT_REPO" rev-parse --short HEAD)" >> "$EXP_DIR/cmd.txt"
job=$(sbatch --parsable -J "lsft-$STAGE" --export=ALL,EXP_DIR="$EXP_DIR",STAGE="$STAGE" "$LSFT_REPO/h200/stage1.sbatch")
echo "submitted $job -> $EXP_DIR ; log: $LSFT_LOGS/lsft-$STAGE-$job.log"
