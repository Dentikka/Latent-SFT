"""Honest-latent variant of a Latent-SFT training jsonl.

In the released data the latent chain replaces only the `<think>` block (`cot`), while the
cleaned prose solution (`cot_answer`) is trained with plain CE and written as text at
inference. Here everything except the final answer goes into the latent chain:

    cot        := think body + "\\n\\n" + prose with its last \\boxed{X} unwrapped to X
    cot_answer := "\\boxed{X}"

Rows whose `cot_answer` has no \\boxed{...} are dropped. The source file is not modified.

Usage: python h200/make_honest_data.py <in.jsonl> <out.jsonl> [--tokenizer PATH]
"""
import argparse
import json
import logging
import statistics
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def last_boxed(text: str) -> Optional[Tuple[int, int, str]]:
    """(start, end, content) of the last balanced \\boxed{...}, or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    if depth:
        return None
    return start, i, text[start + len("\\boxed{"):i - 1]


def strip_think(cot: str) -> str:
    cot = cot.strip()
    if cot.startswith(THINK_OPEN):
        cot = cot[len(THINK_OPEN):]
    if cot.endswith(THINK_CLOSE):
        cot = cot[:-len(THINK_CLOSE)]
    return cot.strip()


def convert(row: dict) -> Optional[dict]:
    box = last_boxed(row["cot_answer"])
    if box is None:
        return None
    start, end, content = box
    prose = (row["cot_answer"][:start] + content + row["cot_answer"][end:]).strip()
    out = dict(row)
    out["cot"] = strip_think(row["cot"]) + "\n\n" + prose
    out["cot_answer"] = "\\boxed{" + content + "}"
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default=None, help="report token lengths with this tokenizer")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    n_in = n_out = n_nobox = n_mismatch = 0
    lens = {"cot_old": [], "answer_old": [], "cot_new": []}
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            n_in += 1
            new = convert(row)
            if new is None:
                n_nobox += 1
                continue
            ref = str(row.get("answer", "")).strip()          # stored as "\boxed{...}"
            if ref and ref != new["cot_answer"] and ref != new["cot_answer"][len("\\boxed{"):-1]:
                n_mismatch += 1
            fout.write(json.dumps(new, ensure_ascii=False) + "\n")
            n_out += 1
            if tok is not None and n_out <= 2000:
                enc = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
                lens["cot_old"].append(enc(strip_think(row["cot"])))
                lens["answer_old"].append(enc(row["cot_answer"]))
                lens["cot_new"].append(enc(new["cot"]))
            if n_out == 1:
                logger.info("example cot_answer: %r", new["cot_answer"])
                logger.info("example cot tail: %r", new["cot"][-200:])

    logger.info("rows in %d, out %d, dropped without \\boxed %d, boxed != `answer` field %d",
                n_in, n_out, n_nobox, n_mismatch)
    for k, v in lens.items():
        if v:
            logger.info("%s tokens (first %d rows): mean %.0f, median %.0f, max %d",
                        k, len(v), statistics.mean(v), statistics.median(v), max(v))


if __name__ == "__main__":
    main()
