#!/bin/bash
# Stage-1 encoder eval on Math-500 with the repo's eval/eval_encoder_hf_batch.py, 1 GPU.
# The encoder compresses the gold `solution` into latents and the frozen decoder answers
# from prompt + latents, so this measures how much the latents carry (an upper bound).
# MAXNEW (default 64) keeps the decoder from re-solving the problem in text.
# Evaluates step 0 (the untrained encoder saved as out/base_model) and every kept epoch,
# via symlinks in <exp>/eval/view laid out as the script expects (checkpoint-<n>/hf).
# Usage: sbatch --export=ALL,EXP=<exp>[,MAXNEW=64][,CKPTS="0 898"] h200/eval_encoder.sbatch
#SBATCH -A fpmi-p10
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=88G
#SBATCH -t 02:00:00
#SBATCH -J lsft-evalenc
#SBATCH -o /home/data/%u/logs/%x-%j.log
set -euo pipefail
source "${LSFT_REPO:-$HOME/projects/repos/Latent-SFT}/h200/env.sh"
: "${EXP:?}"
MAXNEW="${MAXNEW:-64}"
EXP_DIR="$LSFT_EXPS/$EXP"
source "$EXP_DIR/run.env"
VIEW="$EXP_DIR/eval/view"
OUTD="$EXP_DIR/eval/math500-max$MAXNEW"
mkdir -p "$VIEW" "$OUTD"
echo "== job $SLURM_JOB_ID $(date) exp=$EXP maxnew=$MAXNEW repo=$(git -C "$LSFT_REPO" rev-parse --short HEAD)"

FIRST=$(ls -d "$EXP_DIR"/epochs/epoch-* | sort -V | head -1)
if [ ! -e "$VIEW/checkpoint-0" ]; then        # untrained encoder + tokenizer with the compress token
  mkdir -p "$VIEW/checkpoint-0/hf"
  ln -s "$EXP_DIR"/out/base_model/* "$VIEW/checkpoint-0/hf/"
  for f in "$FIRST"/hf/*; do
    case "$(basename "$f")" in model*|config.json|generation_config.json) ;; *) ln -sf "$f" "$VIEW/checkpoint-0/hf/" ;; esac
  done
fi
for d in "$EXP_DIR"/epochs/epoch-*; do
  s="${d##*-step-}"; [ -e "$VIEW/checkpoint-$s" ] || ln -s "$d" "$VIEW/checkpoint-$s"
done

cd "$LSFT_REPO"
for ck in ${CKPTS:-0 $(ls "$EXP_DIR"/epochs | sed 's/.*-step-//' | sort -n)}; do
  if [ -f "$OUTD/${ck}_${MAXNEW}_Math500_result.jsonl" ]; then echo "skip ck $ck (done)"; continue; fi
  echo "== ck $ck $(date)"
  python eval/eval_encoder_hf_batch.py --dataset Math500 \
    --data_path "$LSFT_DATA/Math-500-test.jsonl" --check_point "$ck" \
    --encoder_name_or_path "$VIEW" --decoder_name_or_path "$DECODER" --save_path "$OUTD" \
    --mp_size 1 --batch_size 16 --max_new_tokens "$MAXNEW" \
    --compression_rate "$COMPRESSION" --topk_interpolation "$TOPK"
done
echo "== results"; grep -h Scores "$OUTD"/eval_result_Math500.txt
