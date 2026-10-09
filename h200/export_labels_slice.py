"""Upload the first N rows of a Stage-2 train set with their soft labels to S3 (CPU only).

For diagnostics off the cluster (h200/diag_exit_tf.py): writes rows.jsonl and labels.pt
(torch list of (probs, indices) per row, in train order) under s3://iii-storage2/<prefix>/.
Credentials from the environment (~/.s3-cod.env), endpoint from S3_ENDPOINT.

Usage: python h200/export_labels_slice.py --labels_dir <soft-labels> --train <train.jsonl>
           --n 256 --prefix shveikinda/h200/<series>/<exp>/diag/train-slice
"""
import argparse
import glob
import json
import os
import tempfile

import boto3
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.labels_dir, "batch_*.pt")),
                   key=lambda p: int(os.path.basename(p).split("_")[1]))
    labels = []
    for p in files:
        labels.extend(torch.load(p, map_location="cpu"))
        if len(labels) >= args.n:
            break
    labels = labels[:args.n]
    rows = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == args.n:
                break
    # counts must match the cot length the labels were built from (ceil(len / r) is checked in
    # labels.sbatch); here only a cheap sanity check that both lists line up
    assert len(rows) == len(labels) == args.n, (len(rows), len(labels))
    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT", "https://s3.cod.phystech.edu"))
    with tempfile.TemporaryDirectory() as d:
        torch.save(labels, os.path.join(d, "labels.pt"))
        with open(os.path.join(d, "rows.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"problem": r["problem"], "cot_answer": r["cot_answer"]}, ensure_ascii=False) + "\n")
        for name in ("labels.pt", "rows.jsonl"):
            s3.upload_file(os.path.join(d, name), "iii-storage2", f"{args.prefix.rstrip('/')}/{name}")
    k = [int(p.shape[0]) for p, _ in labels]
    print(f"EXPORTED {args.n} rows from {len(files)} label files; latents per row: "
          f"min {min(k)} median {sorted(k)[len(k) // 2]} max {max(k)} -> s3://iii-storage2/{args.prefix}")


if __name__ == "__main__":
    main()
