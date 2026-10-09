#!/bin/bash
# Alibaba side of the cross-server Stage-2 eval (h200/eval_dispatch.py). Runs in tmux on the login
# node, holds no card: every EVERY seconds, if we have no eval job and a card is obtainable now
# (free, or held by a brain-long job past its 30-min protection, which a brain job preempts), it
# claims the next epoch on the S3 board and submits h200/alibaba/eval_epoch.sbatch. A job still
# pending after PENDING_MAX seconds is cancelled and its claim released.
# Usage: SITE_ENV=<site.env> bash h200/alibaba/poll.sh
# DISPATCH_ARGS in site.env (e.g. "--ns eval-sys4064 --prompt system --max_latent 4064") selects the
# eval protocol; eval_epoch.sbatch passes the same.
set -uo pipefail
: "${SITE_ENV:?}"
. "$SITE_ENV"
set -a; . "$HOME/.s3-cod.env"; set +a
export PYTHONPATH="$PYLIB"
EVERY=${EVERY:-600}; PENDING_MAX=${PENDING_MAX:-2100}; TOTAL=${TOTAL_CARDS:-10}
D="$REPO/h200/eval_dispatch.py"
py() { apptainer exec --bind /bmcp_lvm_fs "$SIF" python3 "$D" "$@" --base "$BASE" ${DISPATCH_ARGS:-} 2>/dev/null; }

obtainable() {   # prints "free preemptible"
  squeue -h -t R -o "%P|%M|%b" | awk -F'|' -v total="$TOTAL" '
    { n = split($3, g, ":"); used += g[n]
      if ($1 == "brain-long") { m = split($2, t, ":"); mins = (index($2, "-") || m == 3) ? 999 : t[1]
                                if (mins >= 30) pre++ } }
    END { printf "%d %d\n", total - used, pre }'
}

while true; do
  job=$(squeue -u "$USER" -h -n lsft-eval -o "%i|%T|%k|%V" | head -1)
  if [ -n "$job" ]; then
    IFS='|' read -r jid state ep sub <<<"$job"
    age=$(( $(date +%s) - $(date -d "$sub" +%s) ))
    if [ "$state" = PENDING ] && [ "$age" -gt "$PENDING_MAX" ]; then
      echo "$(date '+%F %T') job $jid (epoch $ep) pending ${age}s: cancel + release"
      scancel "$jid"; py release --epoch "$ep"
    fi
  else
    read -r free pre <<<"$(obtainable)"
    if [ "$free" -gt 0 ] || [ "$pre" -gt 0 ]; then
      ep=$(py claim --site alibaba)
      if [ -n "$ep" ]; then
        jid=$(sbatch --parsable -p brain --comment="$ep" --export=ALL,EP="$ep",SITE_ENV="$SITE_ENV" \
              -o "$WORK/logs/eval-%j.log" "$REPO/h200/alibaba/eval_epoch.sbatch")
        echo "$(date '+%F %T') claimed epoch $ep -> job $jid (free $free, preemptible $pre)"
      fi
    fi
  fi
  sleep "$EVERY"
done
