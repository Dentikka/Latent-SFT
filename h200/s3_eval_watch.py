"""Watch S3 for new Stage-2 epoch adapters and run the honest eval on each (another server).

The Stage-2 job mirrors `<exp>/epochs/epoch-K-step-N/lora_adapter/` to S3 (h200/s3_sync.py).
Each pass this script takes the newest epoch not yet evaluated, downloads its adapter, and runs
h200/eval_forced_latent.py on the initial model + adapter: the model writes its own latents,
then `\\boxed` is forced. One line per epoch goes to OUT/curve.jsonl. Newest-first, so a slow
eval skips epochs instead of falling behind.

Usage: python h200/s3_eval_watch.py --prefix shveikinda/h200/<series>/<exp>/epochs \\
           --model INIT_DIR --data Math-500-test.jsonl --out OUT_DIR [--n 200] [--every 600]
           [--init_prefix shveikinda/h200/<series>/<union-exp>/stage2-init] [--until_epoch 70]
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time

import boto3

logger = logging.getLogger(__name__)
BUCKET = "iii-storage2"
EPOCH = re.compile(r"epoch-(\d+)-step-(\d+)/lora_adapter/")


def list_keys(s3, prefix: str) -> list:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


def download(s3, prefix: str, dst: str) -> None:
    for key in list_keys(s3, prefix):
        path = os.path.join(dst, os.path.relpath(key, prefix))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            s3.download_file(BUCKET, key, path)


def epochs_ready(s3, prefix: str) -> dict:
    """epoch -> adapter prefix, for adapters whose weights and config are both uploaded."""
    found = {}  # epoch -> (adapter prefix, file names)
    for key in list_keys(s3, prefix):
        m = EPOCH.search(key)
        if m:
            found.setdefault(int(m.group(1)), (key[:m.end()], set()))[1].add(os.path.basename(key))
    return {ep: pre for ep, (pre, names) in found.items()
            if "adapter_config.json" in names and any(n.startswith("adapter_model") for n in names)}


def pick_gpu(allowed: list, min_free_gb: float):
    """Highest-index allowed GPU with at least min_free_gb free memory now, else None."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    free = {int(i): int(m) / 1024 for i, m in (line.split(",") for line in out.strip().splitlines())}
    ok = [i for i in allowed if free.get(i, 0) >= min_free_gb]
    return max(ok) if ok else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init_prefix", default=None, help="download --model from here if missing")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--until_epoch", type=int, default=70)
    ap.add_argument("--gpus", default="", help="allowed GPU indices, e.g. 0,1,2,4,5,6,7 (shared server)")
    ap.add_argument("--min_free_gb", type=float, default=45.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--answer_prefix", default=None,
                    help="forced answer start after </think> (eval_forced_latent default if unset); "
                         "for a model trained on '</think>\boxed{X}' use '\boxed'")
    args = ap.parse_args()
    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT", "https://s3.cod.phystech.edu"))
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(os.path.join(args.model, "config.json")):
        logger.info("downloading the initial model from s3://%s/%s", BUCKET, args.init_prefix)
        download(s3, args.init_prefix, args.model)
    curve = os.path.join(args.out, "curve.jsonl")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    while True:
        done = set()
        if os.path.exists(curve):
            done = {json.loads(line)["epoch"] for line in open(curve, encoding="utf-8")}
        todo = {e: p for e, p in epochs_ready(s3, args.prefix).items() if e not in done}
        if todo:
            ep = max(todo)
            ad = os.path.join(args.out, "adapters", f"epoch-{ep}")
            download(s3, todo[ep], ad)
            ev = os.path.join(args.out, f"epoch-{ep}")
            env = dict(os.environ)
            if args.gpus:
                gpu = pick_gpu([int(g) for g in args.gpus.split(",")], args.min_free_gb)
                if gpu is None:
                    logger.info("epoch %d ready, no allowed GPU with %.0f GB free; waiting", ep, args.min_free_gb)
                    time.sleep(args.every)
                    continue
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            logger.info("epoch %d: eval on %d problems, GPU %s", ep, args.n, env.get("CUDA_VISIBLE_DEVICES"))
            subprocess.run([sys.executable, os.path.join(repo, "h200", "eval_forced_latent.py"),
                            "--model", args.model, "--lora", ad, "--data", args.data, "--out", ev,
                            "--arms", "forced", "--n", str(args.n), "--temperature", "0.6",
                            "--top_p", "0.95", "--samples", "1", "--batch", str(args.batch)]
                           + (["--prefix", args.answer_prefix] if args.answer_prefix is not None else []),
                           check=True, env=env)
            s = json.load(open(os.path.join(ev, "summary.json"), encoding="utf-8"))
            row = {"epoch": ep, "forced": s["forced"], "prefix": s["prefix"], "mean_latent": s.get("mean_latent"),
                   "not_exited": s.get("not_exited"), "n": s["n"]}
            with open(curve, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            logger.info("CURVE %s", json.dumps(row))
            if ep >= args.until_epoch:
                break
            continue
        time.sleep(args.every)


if __name__ == "__main__":
    main()
