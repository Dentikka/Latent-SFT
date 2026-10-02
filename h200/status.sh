#!/bin/bash
# Usage: bash h200/status.sh <exp-name>
# One screen of status for a chained run: Slurm jobs, progress, per-token CE trend across
# all job logs of the exp (logged loss / gradient accumulation, see repro notes), GPUs,
# kept epoch weights and disk.
set -uo pipefail
source "$(dirname "$0")/env.sh"
EXP="${1:?exp name, e.g. 1002-s1-encoder-honest}"
EXP_DIR="$LSFT_EXPS/$EXP"
[ -d "$EXP_DIR" ] || { echo "no such exp: $EXP_DIR"; exit 1; }
source "$EXP_DIR/run.env"
GA=$(( GLOBAL_BATCH / (2 * PER_DEVICE_BS) ))

echo "== jobs"; squeue -u "$USER" -o "%.8i %.6P %.14j %.3t %.10M %R"
LOGS=$(grep -l "exp=$EXP_DIR" "$LSFT_LOGS"/lsft-*.log 2>/dev/null | sort -t- -k3 -n)
echo "== logs: $(echo $LOGS | wc -w) ($(for l in $LOGS; do basename "$l" .log | sed 's/.*-//'; done | tr '\n' ' '))"
LAST=$(echo "$LOGS" | tail -1)
[ -n "$LAST" ] && echo "progress: $(grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' "$LAST" | tail -1)"
for f in DONE STALLED; do [ -f "$EXP_DIR/$f" ] && echo "!! $f"; done

echo "== CE per token (logged loss / $GA), by 0.05 epoch"
cat $LOGS 2>/dev/null | python -c "
import re, sys, collections
b = collections.OrderedDict()
for m in re.finditer(r\"\{'loss': ([0-9.]+), 'grad_norm': ([0-9.e+]+), 'learning_rate': ([0-9.e-]+), 'epoch': ([0-9.]+)\}\", sys.stdin.read()):
    loss, gn, lr, ep = map(float, m.groups())
    b.setdefault(round(ep // 0.05 * 0.05, 2), []).append((loss / $GA, lr))
for ep, v in b.items():
    print(f'  epoch {ep:4.2f}: CE {sum(x for x, _ in v) / len(v):5.2f}  lr {v[-1][1]:.1e}  (n={len(v)})')
"

echo "== GPUs (last minute)"; tail -2 "$EXP_DIR/gpumem.log" 2>/dev/null
echo "== kept epochs"; ls "$EXP_DIR/epochs" 2>/dev/null || echo "  none yet"
echo "== checkpoints"; ls -d "$EXP_DIR"/out/checkpoint-* 2>/dev/null | xargs -n1 basename 2>/dev/null
echo "== disk: exp $(du -sh "$EXP_DIR" 2>/dev/null | cut -f1), /home/data free $(df -h /home/data | awk 'NR==2{print $4}')"
