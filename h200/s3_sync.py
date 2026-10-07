"""Mirror a local directory tree to S3 (idempotent: an object of the same size is skipped).

Credentials come from the environment (`~/.s3-cod.env`: AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY); the endpoint from S3_ENDPOINT, default the lab's S3.

  --include SUB   only files whose relative path contains SUB (repeatable), e.g. lora_adapter
  --every N       repeat every N seconds until --until FILE exists (then one last pass)

Usage: python h200/s3_sync.py SRC_DIR PREFIX [--include lora_adapter] [--every 600 --until EXP/DONE]
"""
import argparse
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
BUCKET = "iii-storage2"


def remote_size(s3, key: str) -> int:
    try:
        return s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    except ClientError:
        return -1


def sync_once(s3, src: str, prefix: str, include: list) -> int:
    sent = 0
    for root, _, files in os.walk(src):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src).replace(os.sep, "/")
            if include and not any(sub in rel for sub in include):
                continue
            key = f"{prefix.rstrip('/')}/{rel}"
            size = os.path.getsize(path)
            if remote_size(s3, key) == size:
                continue
            s3.upload_file(path, BUCKET, key)
            sent += 1
            logger.info("uploaded %s (%.1f MB)", key, size / 2**20)
    return sent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("prefix")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--every", type=int, default=0)
    ap.add_argument("--until", default=None)
    args = ap.parse_args()
    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT", "https://s3.cod.phystech.edu"))
    while True:
        done = bool(args.until) and os.path.exists(args.until)
        if os.path.isdir(args.src):
            n = sync_once(s3, args.src, args.prefix, args.include)
            logger.info("sync pass: %d new files -> s3://%s/%s", n, BUCKET, args.prefix)
        if not args.every or done:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
