#!/bin/bash
# Stage-1 step (encoder | decoder | union) on 2 x H200, submitted by h200/submit.sh.
# The queue caps a job at 8 h, so a long step runs as a chain: each job first queues
# its successor (afterany) and the successor resumes from the newest checkpoint.
# The chain stops on DONE, on a job that made no new checkpoint, or at MAX_CHAIN.
#SBATCH -A fpmi-p10
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=176G
#SBATCH -t 08:00:00
#SBATCH -J lsft-s1
#SBATCH -o /home/data/%u/logs/%x-%j.log
set -euo pipefail
source "${LSFT_REPO:-$HOME/projects/repos/Latent-SFT}/h200/env.sh"
: "${EXP_DIR:?}" "${STAGE:?}"
source "$EXP_DIR/run.env"
cd "$LSFT_REPO"
echo "== job $SLURM_JOB_ID $(date) stage=$STAGE exp=$EXP_DIR repo=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "CUDA_HOME=${CUDA_HOME:-unset} nvcc=$("${CUDA_HOME:-/nonexistent}/bin/nvcc" --version 2>/dev/null | grep -o "release [0-9.]*" || echo none)"

OUT="$EXP_DIR/out"
[ -f "$EXP_DIR/DONE" ] && { echo "DONE already, nothing to do"; exit 0; }
last_ckpt() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1; }
if [ -z "${MAX_STEPS:-}" ]; then
  n=$(( $(cat "$EXP_DIR/chain_count" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "$EXP_DIR/chain_count"
  if [ -f "$EXP_DIR/ckpt_at_start" ] && [ "$(cat "$EXP_DIR/ckpt_at_start")" = "$(last_ckpt)" ]; then
    echo "STALLED: previous job left no new checkpoint ($(last_ckpt)); stopping the chain" | tee "$EXP_DIR/STALLED"
    exit 1
  fi
  last_ckpt > "$EXP_DIR/ckpt_at_start" || true
  if [ "$n" -lt "${MAX_CHAIN:-15}" ]; then
    next=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" -J "$SLURM_JOB_NAME" \
           --export=ALL,EXP_DIR="$EXP_DIR",STAGE="$STAGE" "$LSFT_REPO/h200/stage1.sbatch")
    echo "queued successor $next (chain $n/${MAX_CHAIN:-15})"
  fi
fi

NGPU=2
GA=$(( GLOBAL_BATCH / (NGPU * PER_DEVICE_BS) ))
REPORT=none
if [ -z "${MAX_STEPS:-}" ]; then
  REPORT=wandb
  export WANDB_RUN_ID="$(basename "$EXP_DIR")" WANDB_RESUME=allow WANDB_DIR="$EXP_DIR"
  grep -q wandb-radfan "$HOME/.netrc" 2>/dev/null || export WANDB_MODE=offline
  export LSFT_KEEP_EPOCHS_DIR="$EXP_DIR/epochs"   # end-of-epoch weights survive rotation
fi
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 >> "$EXP_DIR/gpumem.log" &

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run --standalone --nproc_per_node=$NGPU \
  --master_port=$(( 20000 + SLURM_JOB_ID % 20000 )) \
  "script/run_distill_stage1_${STAGE}.py" \
  --encoder_name_or_path "$ENCODER" --decoder_name_or_path "$DECODER" \
  --bfloat16 True --use_flash_attention_2 False --topk_interpolation "$TOPK" \
  --train_data_path "$LSFT_TRAIN" --compression_rate "$COMPRESSION" \
  --lora_tune True --lora_rank 64 --lora_dropout 0.1 \
  --deepspeed config_zero1.json --no_remove_unused_columns \
  --learning_rate "$LR" --warmup_ratio 0.05 --weight_decay 0.01 --lr_scheduler_type cosine \
  --num_train_epochs "$EPOCHS" ${MAX_STEPS:+--max_steps $MAX_STEPS} --bf16 \
  --per_device_train_batch_size "$PER_DEVICE_BS" --gradient_accumulation_steps "$GA" \
  --dataloader_drop_last False --dataloader_num_workers 4 --dataloader_prefetch_factor 16 \
  --dataloader_pin_memory True --logging_steps 1 --gradient_checkpointing "${GRAD_CKPT:-True}" \
  --save_strategy "$SAVE_STRATEGY" ${SAVE_STEPS:+--save_steps $SAVE_STEPS} \
  --save_total_limit "$SAVE_TOTAL_LIMIT" --seed "${SEED:-42}" \
  --report_to "$REPORT" --run_name "$(basename "$EXP_DIR")" \
  --overwrite_output_dir --output_dir "$OUT" --resume_from_checkpoint auto

touch "$EXP_DIR/DONE"
echo "== finished $(date)"
# Hand over to the next step: NEXT="<stage> <preset> <exp> [KEY=VALUE ...]" in run.env
# (re-read by every link, so it can be appended while the chain runs).
if [ -n "${NEXT:-}" ]; then
  [ -n "${next:-}" ] && scancel "$next" && echo "cancelled successor $next"
  echo "== next step: $NEXT"
  bash "$LSFT_REPO/h200/submit.sh" $NEXT
fi
