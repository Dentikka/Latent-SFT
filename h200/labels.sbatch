#!/bin/bash
# After a Stage-1 union run: (1) soft labels for the whole train set from the union's encoder
# (repo generator, latent top-K picked as set by LSFT_LATENT_READOUT in the run's run.env),
# (2) the union's decoder LoRA merged into its decoder base = the Stage-2 initial model.
# Outputs go to <union exp>/soft-labels/ and <union exp>/stage2-init/; LABELS_DONE marks success.
# Usage: sbatch --export=ALL,EXP=<union-exp> [--dependency=afterok:<last union job>] h200/labels.sbatch
#SBATCH -A fpmi-p10
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=176G
#SBATCH -t 08:00:00
#SBATCH -J lsft-labels
#SBATCH -o /home/data/%u/logs/%x-%j.log
set -euo pipefail
source "${LSFT_REPO:-$HOME/projects/repos/Latent-SFT}/h200/env.sh"
: "${EXP:?}"
EXP_DIR="$LSFT_EXPS/$EXP"
source "$EXP_DIR/run.env"
cd "$LSFT_REPO"
echo "== job $SLURM_JOB_ID $(date) exp=$EXP repo=$(git rev-parse --short HEAD) readout=${LSFT_LATENT_READOUT:-embed}"
[ -f "$EXP_DIR/DONE" ] || { echo "union run not DONE: $EXP_DIR"; exit 1; }
[ -f "$EXP_DIR/LABELS_DONE" ] && { echo "LABELS_DONE already"; exit 0; }
LORA="$(ls -d "$EXP_DIR"/epochs/epoch-*/lora_adapter | sort -V | tail -1)"
echo "encoder=$ENCODER decoder=$DECODER lora=$LORA data=$LSFT_TRAIN"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 >> "$EXP_DIR/gpumem-labels.log" &
export LSFT_LABELS_NUMPY=1   # worker results via pipe, not /dev/shm (job 360 aborted on shm unlink)

if [ ! -f "$EXP_DIR/soft-labels/COMPLETE" ]; then
  rm -rf "$EXP_DIR/soft-labels"
  python generate_latent_soft_label_lora_batch.py --encoder_model_path "$ENCODER" \
    --decoder_model_path "$DECODER" --lora_path "$LORA" --save_path "$EXP_DIR/soft-labels" \
    --data_path "$LSFT_TRAIN" --mp_size 2 --batch_size 16 \
    --compression_rate "$COMPRESSION" --topk_interpolation "$TOPK"
  touch "$EXP_DIR/soft-labels/COMPLETE"
fi
python merge_lora.py --base_model_path "$DECODER" --lora_path "$LORA/decoder_weight" \
  --output_path "$EXP_DIR/stage2-init" --output_subdir "" --device_map cpu
python - "$EXP_DIR" "$LSFT_TRAIN" <<'PY'
import glob, os, sys, torch
exp, train = sys.argv[1], sys.argv[2]
n_rows = sum(1 for _ in open(train, encoding="utf-8"))
files = sorted(glob.glob(os.path.join(exp, "soft-labels", "batch_*.pt")), key=lambda p: int(os.path.basename(p).split("_")[1]))
first = torch.load(files[0], map_location="cpu")
n = sum(int(os.path.basename(p).split("_")[2].split(".")[0]) - int(os.path.basename(p).split("_")[1]) for p in files)
probs, idx = first[0]
print(f"labels: {n} rows in {len(files)} files (train rows {n_rows}); first: {tuple(probs.shape)} latents, "
      f"top-1 weight {probs[:, 0].float().mean():.3f}, N_eff {(1 / (probs.float() ** 2).sum(-1)).mean():.2f}")
assert n == n_rows, "label rows != train rows"
PY
touch "$EXP_DIR/LABELS_DONE"
echo "== finished $(date)"
