#!/bin/bash
# Base model for self-distillation (presets/s2-sd.env): the merged Stage-2 epoch-40 weights (hard
# links from the Latent-GRPO init, which holds them after the epoch hf/ dirs are cleaned) with the
# Stage-2 tokenizer (base chat template; the training code appends '<think>' itself). Seconds, no GPU.
# Usage: bash h200/make_sd_base.sh [SRC=<merged model>] [TOK=<stage2-init>]
set -euo pipefail
source "$(dirname "$0")/env.sh"
SRC="${SRC:-$LSFT_MODELS/lgrpo-init-s2-ep40}"
TOK="${TOK:-$LSFT_EXPS/1006-s1-union-lmhead/stage2-init}"
DST="${DST:-$LSFT_MODELS/qwen-s2-ep40}"
[ -f "$SRC/config.json" ] && [ -f "$TOK/tokenizer_config.json" ] || { echo "missing $SRC or $TOK"; exit 1; }
mkdir -p "$DST"
for f in "$SRC"/*.safetensors; do ln -f "$f" "$DST/"; done
for n in config.json generation_config.json model.safetensors.index.json; do
  [ -f "$SRC/$n" ] && cp -f "$SRC/$n" "$DST/"
done
for n in tokenizer_config.json tokenizer.json vocab.json merges.txt special_tokens_map.json added_tokens.json chat_template.jinja; do
  [ -f "$TOK/$n" ] && cp -f "$TOK/$n" "$DST/"
done
python - "$SRC" "$DST" <<'PY'
import sys
from transformers import AutoTokenizer
src, dst = sys.argv[1:]
a, b = AutoTokenizer.from_pretrained(src), AutoTokenizer.from_pretrained(dst)
assert a.get_vocab() == b.get_vocab(), "vocabularies differ"
m = [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}]
t = b.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
assert not t.rstrip().endswith("<think>"), repr(t[-60:])
print("SD_BASE_OK", dst, "| vocab", len(b), "| prompt tail", repr(t[-30:]))
PY
