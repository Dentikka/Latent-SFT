# Paths for the FPMI H200 cluster (samcs-gpu). Sourced by every h200/ script.
# Heavy data lives on /home/data (3.5 TB, no backup); code stays in $HOME.
export LSFT_REPO="${LSFT_REPO:-$HOME/projects/repos/Latent-SFT}"
export LSFT_ROOT="${LSFT_ROOT:-/home/data/$USER}"
export LSFT_ENV="$LSFT_ROOT/envs/latent-sft"
export LSFT_MODELS="$LSFT_ROOT/models"
export LSFT_DATA="$LSFT_ROOT/data/latent-reasoning"
export LSFT_SERIES="${LSFT_SERIES:-0930-qwen-lsft-repro}"
export LSFT_EXPS="$LSFT_ROOT/projects/latent-sft/exps/$LSFT_SERIES"
export LSFT_LOGS="$LSFT_ROOT/logs"

# Base model prepared by setup.sbatch: Qwen2.5-Math-7B + '<think>' in the generation
# prompt + max_position_embeddings=16384, exactly as in the released
# DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10. The path must contain "qwen":
# src/stage1/data.py picks the prompt format by substring of the model path.
export LSFT_BASE="$LSFT_MODELS/Qwen2.5-Math-7B-think"
export LSFT_TRAIN="$LSFT_DATA/OpenR1-Math-220k-v-train-4k.jsonl"

export HF_HOME="$LSFT_ROOT/hf-cache"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://wandb-radfan.ru}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-sft}"
export PATH="$LSFT_ENV/bin:$PATH"

# DeepSpeed checks op compatibility at import by running `$CUDA_HOME/bin/nvcc -V` and
# dies without a CUDA toolkit, which the node does not have. The pip wheel
# nvidia-cuda-nvcc-cu12 (12.4, matching torch cu124) provides nvcc; nothing is compiled
# (the optimizer is torch AdamW), only the version is read.
if [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc >/dev/null 2>&1; then
  _nvcc_home="$LSFT_ENV/lib/python3.12/site-packages/nvidia/cuda_nvcc"
  if [ -x "$_nvcc_home/bin/nvcc" ]; then export CUDA_HOME="$_nvcc_home"; fi
fi
