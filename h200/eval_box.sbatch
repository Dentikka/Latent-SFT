#!/bin/bash
# Honest-mode Math-500 eval (h200/eval_box.py: forced "\boxed", greedy) of a Stage-1 decoder
# run, 1 GPU: the base decoder (step 0) and every kept epoch, each with and without latents.
# The encoder is the run's own ENCODER from run.env.
# Usage: sbatch --export=ALL,EXP=<decoder-exp>[,TAGS="base epoch-1-step-898"] h200/eval_box.sbatch
#SBATCH -A fpmi-p10
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=88G
#SBATCH -t 02:00:00
#SBATCH -J lsft-evalbox
#SBATCH -o /home/data/%u/logs/%x-%j.log
set -euo pipefail
source "${LSFT_REPO:-$HOME/projects/repos/Latent-SFT}/h200/env.sh"
: "${EXP:?}"
EXP_DIR="$LSFT_EXPS/$EXP"
source "$EXP_DIR/run.env"
OUTD="$EXP_DIR/eval/box"
mkdir -p "$OUTD"
cd "$LSFT_REPO"
echo "== job $SLURM_JOB_ID $(date) exp=$EXP encoder=$ENCODER repo=$(git rev-parse --short HEAD)"

for tag in ${TAGS:-base $(ls "$EXP_DIR"/epochs 2>/dev/null | sort -V)}; do
  if [ "$tag" = base ]; then dec="$DECODER"; else dec="$EXP_DIR/epochs/$tag/hf"; fi
  for mode in latents none; do
    out="$OUTD/$tag-$mode.jsonl"
    [ -f "${out%.jsonl}.summary.json" ] && { echo "skip $tag $mode"; continue; }
    echo "== $tag $mode $(date)"
    python h200/eval_box.py --encoder "$ENCODER" --decoder "$dec" \
      --data "$LSFT_DATA/Math-500-test.jsonl" --out "$out" --mode "$mode" \
      --r "$COMPRESSION" --topk "$TOPK"
  done
done
echo "== results"
python - "$OUTD" <<'EOF'
import glob, json, os, sys
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.summary.json"))):
    s = json.load(open(p))
    print(f"{os.path.basename(p)[:-13]:32s} acc {s['acc']:.3f}  box closed {s['box_closed']:.2f}  latents {s['mean_latents']:.0f}")
EOF
