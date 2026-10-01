# Reproducing Qwen2.5-Math-7B-Latent-SFT-4k-Top10 on the FPMI H200 cluster

Target: `DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10`, the initialisation used by Latent-GRPO
for the high-difficulty track. The cluster is browser-only (Open OnDemand terminal), Slurm
account `fpmi-p10`, queue `gpu`, 2 x H200 NVL, 8 h per job.

1. `sbatch h200/setup.sbatch` — uv env (torch 2.5.1+cu124, transformers 4.52.4, peft 0.15.2,
   deepspeed 0.17.0, flash-attn 2.7.3 wheel), training/eval data, `Qwen/Qwen2.5-Math-7B`,
   the released checkpoint for comparison, and `Qwen2.5-Math-7B-think`: the base model with
   the only two differences found between its config/tokenizer and the released
   checkpoint's (`<think>` appended to the generation prompt, `max_position_embeddings`
   16384).
2. `bash h200/submit.sh encoder smoke <exp>` — 20 steps to measure step time and memory.
3. `bash h200/submit.sh <encoder|decoder|union> <preset> <exp>` — full Stage-1 steps.
   Jobs chain themselves across the 8 h limit and resume via
   `--resume_from_checkpoint auto` (added to all four entrypoints; unset = original behaviour).

Per-user caps (QOS): cpu queue 8 CPUs / 45 GB, gpu queue 32 CPUs / 2 GPUs / 180 GB.
Global batch is kept at 64 (the original 8 GPUs x 1 x 8) by raising gradient accumulation.
