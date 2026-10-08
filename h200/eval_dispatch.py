"""Adaptive per-epoch honest eval of a Stage-2 run across servers, coordinated through S3.

The Stage-2 job uploads each epoch's LoRA adapter to <base>/epochs/. Every server runs a worker;
workers never talk to each other, only to S3:

  <base>/eval/results/epoch-K/{summary.json,records.jsonl}   done
  <base>/eval/claims/epoch-K.json                            {"site", "state", "time"}

Rule: at most ONE eval in the whole system at a time. A claim is active while it is "pending"
(a Slurm job waiting to start) for < PENDING_TTL or "running" for < RUNNING_TTL; a worker starts
nothing while any claim is active. The newest epoch without a result goes first, then gaps.

Subcommands:
  claim --site S            print the epoch claimed for S (or nothing); used by the Slurm poller
  start --epoch K --site S  mark the claim running (the eval job calls it when it starts)
  run --epoch K --site S    eval epoch K here (download init + adapter, eval, upload results)
  release --epoch K         drop a claim (a pending job that was cancelled)
  loop --site S             local worker: when an allowed GPU has --min_free_gb free, claim + run
  curve                     print all results

Env: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (~/.s3-cod.env), S3_ENDPOINT (default the lab S3).
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
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
BUCKET = "iii-storage2"
PENDING_TTL = 40 * 60
RUNNING_TTL = 3 * 3600
EPOCH = re.compile(r"epoch-(\d+)-step-(\d+)/lora_adapter/")


class Board:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT", "https://s3.cod.phystech.edu"))

    def keys(self, prefix: str) -> list:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
            out += [o["Key"] for o in page.get("Contents", [])]
        return out

    def ready(self) -> dict:
        """epoch -> adapter prefix, for adapters whose weights and config are both uploaded."""
        found = {}
        for key in self.keys(f"{self.base}/epochs/"):
            m = EPOCH.search(key)
            if m:
                found.setdefault(int(m.group(1)), (key[:m.end()], set()))[1].add(os.path.basename(key))
        return {e: p for e, (p, names) in found.items()
                if "adapter_config.json" in names and any(n.startswith("adapter_model") for n in names)}

    def done(self) -> set:
        return {int(m.group(1)) for k in self.keys(f"{self.base}/eval/results/")
                if (m := re.search(r"epoch-(\d+)/summary\.json$", k))}

    def claims(self) -> dict:
        out = {}
        for k in self.keys(f"{self.base}/eval/claims/"):
            m = re.search(r"epoch-(\d+)\.json$", k)
            if m:
                out[int(m.group(1))] = json.loads(self.s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        return out

    def active(self, done: set) -> dict:
        now = time.time()
        return {e: c for e, c in self.claims().items() if e not in done and
                now - c["time"] < (RUNNING_TTL if c["state"] == "running" else PENDING_TTL)}

    def put_claim(self, epoch: int, site: str, state: str) -> None:
        body = json.dumps({"site": site, "state": state, "time": time.time()})
        self.s3.put_object(Bucket=BUCKET, Key=f"{self.base}/eval/claims/epoch-{epoch}.json", Body=body)

    def release(self, epoch: int) -> None:
        self.s3.delete_object(Bucket=BUCKET, Key=f"{self.base}/eval/claims/epoch-{epoch}.json")

    def claim(self, site: str):
        done = self.done()
        if self.active(done):
            return None
        todo = sorted(set(self.ready()) - done, reverse=True)
        if not todo:
            return None
        self.put_claim(todo[0], site, "pending")
        return todo[0]

    def download(self, prefix: str, dst: str) -> None:
        for key in self.keys(prefix):
            path = os.path.join(dst, os.path.relpath(key, prefix))
            size = self.s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
            if os.path.exists(path) and os.path.getsize(path) == size:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.s3.download_file(BUCKET, key, path)

    def upload_results(self, epoch: int, src: str) -> None:
        for name in ("summary.json", "records.jsonl"):
            self.s3.upload_file(os.path.join(src, name), BUCKET, f"{self.base}/eval/results/epoch-{epoch}/{name}")

    def curve(self) -> list:
        rows = []
        for e in sorted(self.done()):
            s = json.loads(self.s3.get_object(
                Bucket=BUCKET, Key=f"{self.base}/eval/results/epoch-{e}/summary.json")["Body"].read())
            rows.append({"epoch": e, "forced": s["forced"], "mean_latent": s.get("mean_latent"),
                         "not_exited": s.get("not_exited"), "n": s["n"], "site": s.get("site")})
        return rows


def run_eval(board: Board, args, epoch: int, env=None) -> int:
    """Download init (cached) + the epoch's adapter, eval, upload results. Returns the exit code."""
    if not os.path.exists(os.path.join(args.model, "config.json")):
        logger.info("downloading the initial model to %s", args.model)
        board.download(args.init_prefix, args.model)
    ad = os.path.join(args.work, "adapters", f"epoch-{epoch}")
    board.download(board.ready()[epoch], ad)
    out = os.path.join(args.work, f"epoch-{epoch}")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo, "h200", "eval_forced_latent.py"), "--model", args.model,
           "--lora", ad, "--data", args.data, "--out", out, "--arms", "forced", "--n", str(args.n),
           "--temperature", "0.6", "--top_p", "0.95", "--samples", "1", "--batch", str(args.batch),
           "--prefix", args.answer_prefix]
    rc = subprocess.run(cmd, env=env).returncode
    if rc == 0:
        s = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
        s["site"] = args.site
        json.dump(s, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), indent=1)
        board.upload_results(epoch, out)
        logger.info("RESULT epoch %d: forced %.3f, not exited %s (%s)", epoch, s["forced"], s.get("not_exited"), args.site)
    return rc


def pick_gpu(allowed: list, min_free_gb: float):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    free = {int(i): int(m) / 1024 for i, m in (line.split(",") for line in out.strip().splitlines())}
    ok = [i for i in allowed if free.get(i, 0) >= min_free_gb]
    return max(ok) if ok else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["claim", "start", "run", "release", "loop", "curve"])
    ap.add_argument("--base", required=True, help="S3 prefix of the Stage-2 run, e.g. shveikinda/h200/<series>/<exp>")
    ap.add_argument("--site", default="local")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--model", help="local dir of the initial Stage-2 model")
    ap.add_argument("--init_prefix", help="S3 prefix to download --model from")
    ap.add_argument("--data")
    ap.add_argument("--work", help="local dir for adapters and eval outputs")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--answer_prefix", default="\\boxed")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--min_free_gb", type=float, default=45.0)
    ap.add_argument("--every", type=int, default=600)
    args = ap.parse_args()
    board = Board(args.base)

    if args.cmd == "claim":
        e = board.claim(args.site)
        if e is not None:
            print(e)
    elif args.cmd == "start":
        board.put_claim(args.epoch, args.site, "running")
    elif args.cmd == "release":
        board.release(args.epoch)
    elif args.cmd == "run":
        board.put_claim(args.epoch, args.site, "running")
        sys.exit(run_eval(board, args, args.epoch))
    elif args.cmd == "curve":
        for row in board.curve():
            print(json.dumps(row))
    elif args.cmd == "loop":
        allowed = [int(g) for g in args.gpus.split(",")] if args.gpus else []
        while True:
            gpu = pick_gpu(allowed, args.min_free_gb) if allowed else None
            e = board.claim(args.site) if (gpu is not None or not allowed) else None
            if e is not None:
                board.put_claim(e, args.site, "running")
                env = dict(os.environ)
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                logger.info("epoch %d: eval on GPU %s", e, gpu)
                if run_eval(board, args, e, env) != 0:
                    logger.info("epoch %d: eval failed; releasing the claim", e)
                    board.release(e)
                    time.sleep(args.every)
                continue
            time.sleep(args.every)


if __name__ == "__main__":
    main()
