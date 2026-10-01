#!/bin/bash
# One-off setup on the CPU queue: uv env, data, base model, reference checkpoint.
# Idempotent: rerunning skips whatever is already in place.
#SBATCH -A fpmi-p10
#SBATCH -p cpu
#SBATCH -J lsft-setup
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /home/data/%u/logs/%x-%j.log
set -euo pipefail
source "${LSFT_REPO:-$HOME/projects/repos/Latent-SFT}/h200/env.sh"
mkdir -p "$LSFT_MODELS" "$LSFT_DATA" "$LSFT_EXPS" "$HF_HOME"
echo "== $(date) on $(hostname), repo $(git -C "$LSFT_REPO" rev-parse --short HEAD)"
echo "nvcc: $(command -v nvcc || echo none)"

# ---------------------------------------------------------------- environment
PY="$LSFT_ENV/bin/python"
[ -x "$PY" ] || uv venv --python 3.12 "$LSFT_ENV"
pip_() { uv pip install --python "$PY" "$@"; }
pip_ torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip_ transformers==4.52.4 peft==0.15.2 datasets==3.6.0 accelerate==1.7.0 \
     scikit-learn==1.7.0 wandb tensorboard jsonlines sentencepiece protobuf \
     "huggingface_hub[hf_transfer]" setuptools wheel ninja packaging
DS_BUILD_OPS=0 pip_ --no-build-isolation deepspeed==0.17.0
pip_ "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
# Eval-only deps; their pins may clash, so a failure here must not stop the setup.
pip_ pylatexenc==2.10 latex2sympy2==1.9.1 word2number==1.1 || echo "WARN: eval deps failed"
"$PY" - <<'EOF'
import torch, transformers, peft, deepspeed, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| deepspeed", deepspeed.__version__, "| datasets", datasets.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("WARN flash_attn import:", e)
EOF

# ---------------------------------------------------------------- data + models
export HF_HUB_ENABLE_HF_TRANSFER=1
"$PY" - <<'EOF'
import json, os, shutil, urllib.request
from huggingface_hub import snapshot_download, hf_hub_download
M, D = os.environ["LSFT_MODELS"], os.environ["LSFT_DATA"]
for f in ["OpenR1-Math-220k-v-train-4k.jsonl", "Math-500-test.jsonl", "AIME-2024-test.jsonl",
          "AIME-2025-test.jsonl", "GPQA-test.jsonl"]:
    hf_hub_download("DJCheng/Latent-Reasoning-Data", f, repo_type="dataset", local_dir=D)
base = snapshot_download("Qwen/Qwen2.5-Math-7B", local_dir=f"{M}/Qwen2.5-Math-7B")
ref = snapshot_download("DJCheng/Qwen2.5-Math-7B-Latent-SFT-4k-Top10",
                        local_dir=f"{M}/Qwen2.5-Math-7B-Latent-SFT-4k-Top10")

# Training base: Qwen2.5-Math-7B weights (hard links) + max_position_embeddings=16384 as in the
# released checkpoint's config. The chat template stays the base one: for Qwen,
# src/stage1/data.py appends the '<think>' token ids itself after
# apply_chat_template(add_generation_prompt=True). The '<think>' in the released
# checkpoint's template is for Latent-GRPO rollouts/eval (see the Latent-GRPO README) and
# must NOT be in the training template, or every example starts with '<think><think>'.
dst = os.environ["LSFT_BASE"]
os.makedirs(dst, exist_ok=True)
for name in os.listdir(base):
    s, t = os.path.join(base, name), os.path.join(dst, name)
    if not os.path.isfile(s):
        continue
    if name.endswith(".safetensors"):
        if not os.path.exists(t):
            os.link(s, t)
    else:
        shutil.copy2(s, t)          # small files: always refresh from the base
cfg = json.load(open(f"{dst}/config.json"))
cfg["max_position_embeddings"] = 16384
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
tc = json.load(open(f"{dst}/tokenizer_config.json"))
tpl = open(f"{ref}/chat_template.jinja").read().replace("\r", "")
gen_old, gen_new = "'<|im_start|>assistant\\n' }}", "'<|im_start|>assistant\\n<think>' }}"
assert tpl == tc["chat_template"].replace("\r", "").replace(gen_old, gen_new), \
    "released template differs from base by more than the '<think>' generation prompt"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(dst)
p = tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                            tokenize=False, add_generation_prompt=True)
assert p.endswith("<|im_start|>assistant\n"), p
n = sum(1 for _ in open(f"{D}/OpenR1-Math-220k-v-train-4k.jsonl"))
print(f"OK base={dst} train_rows={n}")
EOF
du -sh "$LSFT_MODELS"/* "$LSFT_DATA" 2>/dev/null
df -h /home/data | tail -1
echo "== SETUP DONE $(date)"
